import random
from vllm import LLM
from transformers import AutoTokenizer
from entropy_chain_local_manager import EntropyGuidedChainLocalManager
from parallel_mcts import (
    MCTSr,
    SelectionPolicy,
    EvaluationStrategy,
    gather_paths,
    pass_rate, 
    visualize_tree, 
    print_tree,
)

import math
import os
import json
from multiprocessing import Process
from filelock import FileLock

from tqdm import tqdm


def select_paths_with_ratio(paths, num_traces=32):
    # Shuffle the paths to ensure random selection order
    random.shuffle(paths)

    selected_paths = []
    remaining_paths = []

    # Traverse the shuffled paths and select the first pass_ratio == 1 path
    for path in paths:
        if path[-1]["pass_ratio"] == 1 and len(selected_paths) == 0:
            selected_paths.append(path)
        else:
            remaining_paths.append(path)

    # Calculate how many additional paths we need
    remaining_num_traces = num_traces - len(selected_paths)

    # Randomly select remaining_num_traces paths from the remaining_paths if possible
    if remaining_num_traces > 0:
        selected_paths.extend(random.sample(remaining_paths, min(
            remaining_num_traces, len(remaining_paths))))

    # Shuffle the selected paths to ensure they are returned in random order
    random.shuffle(selected_paths)
    assert len(
        selected_paths) == num_traces, f"len(selected_paths) = {len(selected_paths)} != num_traces = {num_traces}"

    return selected_paths


def normalize_selected_terminals(paths):
    leaf_orm_value = [path[-1]["value"] for path in paths]
    _sum = sum(leaf_orm_value)
    num = len(leaf_orm_value) - 1
    if num == 0:
        return paths
    else:
        mean = [(_sum - leaf_orm_value[i]) /
                num for i in range(len(leaf_orm_value))]
        orm_normalized = [leaf_orm_value[i] - mean[i]
                          for i in range(len(leaf_orm_value))]
        for i in range(len(orm_normalized)):
            paths[i][-1]["value"] = orm_normalized[i]
        return paths


def parallel_entropy_guided_tree(
    item,
    llm,
    # tokenizer,
    args=None,
    tokenize_fn=None,
    decode_fn=None,
    system_prompt=None,
):
    manager = EntropyGuidedChainLocalManager(
        args=args,
        llm=llm,
        encode_fn=tokenize_fn,
        decode_fn=decode_fn,
        evaluator_urls=args['evaluator_urls'],
        extractor_urls=args['extractor_urls'],
        eos_tokens_set=args['eos_tokens'],
    )

    result = manager.process_single_item(item, args)
    paths = result["paths"]
    if args["training_type"] == "general":
        raw_avg_reward = result["raw_avg_reward"]
        return paths, raw_avg_reward
    return paths


def process_single_data_for_each_gpu(
    data_batch, gpu_id, tokenizer_path, eos_tokens, tokenize_fn, system_prompt=None
):
    '''
    仅用作评测本地 vllm 推理性能，不进入 RL 训练
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model=tokenizer_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        seed=3407,
    )
    args = {
        "temperature": 1.2,
        "top_p": 0.9,
        "m": 6,
        "n": 2,
        "l": 0,
        "t": 2,
        "eos_tokens": eos_tokens,
        "use_pure_binary": True, # 是否只使用二分类奖励，若为 False 则使用 rm 评分作为奖励
        "entropy_rm_urls": ["http://172.18.80.30:8000/encode"],
        "num_traces": 30, # 每轮迭代的轨迹数量
        "use_pure_RM" : True, # 是否只使用 rm 评分作为奖励，若为 False 则使用综合评分（可能包含 rm 评分、链路奖励等）作为奖励
        "use_orm_reward" : False, # 是否使用 openrm 评分作为奖励的一部分
        "use_chain_reward" : False, # 是否使用链路奖励作为奖励的一部分
        "step_level_norm" : False, # 是否在每一步进行奖励归一化，若为 False 则只在最终选择的轨迹上进行奖励归一化
        "use_state_value_reward" : False,
        "use_value_only" : True,
        "balance_ratio": 0.2,
        "average_one_generation" : False,
        "advantage_mix_allancestor" : False,
        "use_weighted_value": False,
        "use_all_terminals": True,
        "a": 0.5,
        "b": 0.5,
        "generate_max_len" : 4096,
        "weighted_value_style": "sqrt",
        "overall_norm_style": "token",
        "inner_repetition_penalty" : False,
        "use_diverse_sampling" : False,
        "training_type": "general",
        "system_prompt": system_prompt
    }

    manager = EntropyGuidedChainLocalManager(
        args=args,
        llm=llm,
        encode_fn=tokenize_fn,
        decode_fn=decode_fn,
        eos_tokens_set=args['eos_tokens'],
    )

    score_list = []
    for data in tqdm(data_batch, desc=f"GPU {gpu_id} progress"):
        item = {
            "problem": data["Question"],
            "golden_answer": data["Answer"],
        }
        result = manager.process_single_item(item, args)
        score = result['raw_avg_reward']
        score_list.append(score)
        avg_score = sum(score_list) / len(score_list)
        print(f"score: {score:.4f}, avg_score: {avg_score:.4f}")
    print(f"GPU {gpu_id} finished processing. Scores: {avg_score:.4f}")

def process_single_data_for_each_gpu_mcts(
    data_batch, gpu_id, tokenizer_path, tokenize_fn, detokenize_fn, eos_tokens, system_prompt=None
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    args = {
        "temperature": 0.8, 
        "top_p": 0.9, 
        "eos_tokens": eos_tokens,
        "max_depth": 40, # 最大树深
        "max_node_per_depth": 18, # 每层最大节点数
        "max_nodes": 256, # 最大节点数
        "max_children": 4, # 每个节点最大孩子数 
        "min_children": 4,  # 每个节点最少孩子数, 当使用shallow_enwide时
        "shallow_enwide": False, # "Shallow and wide"（浅而宽）一种广度优先倾向的MCTS变体
        "prompt_key": "problem", 
        "answer_key": "golden_answer", 
        "backbone": "qwen", 
        "use_all_terminals": False,
        "path_num": 30, # 最终选择的轨迹数量, 用于评测
        "pass_k": 30, # 无用参数，功能同 path_num
        "concurrent_num": 4, # 每次选择扩展的节点数量，也是每次并行推理的数量
        "first_token_temperature": 0, # 生成子节点时, 第一个 token 的 temperature, 用于增加多样性
        "exploration_constant": 0.5, # (node.value+offset+1e-8)/2 + self.exploration_constant * math.sqrt(math.log(node.parent.visits + 1) / (node.visits + self.epsilon))
        "backprop": True, # 是否进行奖励反向传播更新节点价值
        "look_ahead": 0,  # 无用参数
        "prompt_max_len": 4096,
        "max_token_num": 4096,
        "max_time_use": 360, 
        "step_level_norm": False,
        "random_pick": False, # 每次选择扩展节点时，是否随机选择
        "use_orm_reward": False,
        "select_correct_leaf": False, # 是否至少选择一个正确的叶子节点
        "use_chain_reward":  False,
        "use_state_value_reward": False,
        "use_value_only": False,
        "use_pure_RM": False,
        "use_pure_binary": True,
        "average_one_generation": False,
        "advantage_mix_allancestor": False,
        "use_weighted_value": False,
        "a": 0.5,
        "b": 0.5,
        "use_api_generation": False,
        "enable_info": False,
        "check_step_validity": False,
        "evalation_strategy": EvaluationStrategy.MODEL_EVALUATION
    }
    if not args["use_api_generation"]:
        llm = LLM(
            model=tokenizer_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.75,
            seed=3407,
        )
    else:
        llm = None
    
    score_list = []
    for data in tqdm(data_batch, desc=f"GPU {gpu_id} progress"):
        problem = data["Question"]
        answer = data["Answer"]

        mcts = MCTSr(
            temperature=args["temperature"],
            top_p=args["top_p"],
            problem=problem,
            golden_answer=answer,
            max_nodes=args["max_nodes"],
            max_node_per_depth = args["max_node_per_depth"],
            max_children=args["max_children"],
            min_children=args["min_children"],
            shallow_enwide = args["shallow_enwide"],
            max_depth=args["max_depth"],
            random_pick = args["random_pick"],
            exploration_constant=args["exploration_constant"],
            selection_policy=SelectionPolicy.GREEDY,
            backbone=args["backbone"],
            pass_k=args["pass_k"],
            backprop=args["backprop"],
            first_token_temperature=args["first_token_temperature"],
            look_ahead=args["look_ahead"],
            llms=[llm],
            tokenizer=tokenizer,
            tokenize_fn = tokenize_fn,
            detokenize_fn = detokenize_fn,
            concurrent_num=args["concurrent_num"],
            path_num = args["path_num"],
            prompt_max_len = args["prompt_max_len"],
            max_token_num = args["max_token_num"],
            max_time_use = args["max_time_use"],
            step_level_norm = args["step_level_norm"],
            use_weighted_value = args["use_weighted_value"],
            use_orm_reward = args["use_orm_reward"],
            select_correct_leaf = args["select_correct_leaf"],
            use_chain_reward = args["use_chain_reward"],
            use_state_value_reward = args["use_state_value_reward"],
            use_value_only = args["use_value_only"],
            use_pure_RM = args["use_pure_RM"],
            use_pure_binary = args["use_pure_binary"],
            system_prompt=system_prompt,
            average_one_generation = args["average_one_generation"],
            a = args["a"],
            b = args["b"],
            eos_tokens_set=args['eos_tokens'],
            # custom
            use_api_generation=args["use_api_generation"],
            enable_info=args["enable_info"],
            evaluation_strategy=args["evalation_strategy"],
            check_step_validity=args["check_step_validity"]
        )
        
        mcts.run()
        paths = gather_paths(
            mcts.root,
            mcts.selected_terminals,
            args["path_num"],
            use_orm_reward=mcts.use_orm_reward,
            use_chain_reward=mcts.use_chain_reward,
            step_level_norm=mcts.step_level_norm,
            use_state_value_reward=mcts.use_state_value_reward,
            use_value_only=mcts.use_value_only,
            average_one_generation=mcts.average_one_generation,
            advantage_mix_allancestor=args["advantage_mix_allancestor"]
        )
        print(f"Gathered {len(paths)} paths.")
        score = pass_rate(paths)
        score_list.append(score)
        avg_score = sum(score_list) / len(score_list)
        print(f"score: {score:.4f}, avg_score: {avg_score:.4f}, sample_num: {len(score_list)}")
        # visualize_tree(mcts.root)
        # print_tree(mcts.root)
    print(f"GPU {gpu_id} finished processing. Scores: {avg_score:.4f}")


if __name__ == '__main__':
    MODEL_PATH = "/home/linziyong/Desktop/Model/Qwen2.5-3B-Instruct"
    EOS_TOKENS = [151643]
    system_prompt_1 = 'please reason step by step with steps separated by "\n\n", and put the index of the correct answer within \\boxed{{}}.'
    system_prompt_2 = r"""
You are a logical reasoning expert. Your task is to analyze the given problem step by step, and output your reasoning in a structured format.

## INSTRUCTIONS
1. **Reasoning Process**: Think through the problem step by step. Each step should be a clear logical progression.
2. **Output Format**: Each reasoning step must be enclosed in `<step>` and `</step>` tags.
3. **Step Structure**: Within each step, use `<premise>` and `</premise>` tags for individual premises, and `<conclusion>` and `</conclusion>` tags for the conclusion of that step.
4. **Step Separation**: Each complete step block must be separated by exactly TWO newlines (\n\n).
5. **Final Answer**: At the end, after all steps, provide the correct answer index within `\boxed{}`.

## OUTPUT FORMAT SPECIFICATION
Your output should have this exact structure:

<step>
<premise>Your first premise or observation for this step</premise>
<premise>Your second premise (if applicable for this step)</premise>
<conclusion>The logical conclusion drawn from these premises in this step</conclusion>
</step>

<step>
<premise>First premise of the second step</premise>
<conclusion>Conclusion of the second step</conclusion>
</step>

[More steps as needed...]

\boxed{answer_index}

## TAG RULES
- **Each complete reasoning unit** must be wrapped in `<step>` and `</step>` tags
- **Premises** within a step go in `<premise>` and `</premise>` tags
- **Conclusion** of a step goes in `<conclusion>` and `</conclusion>` tags
- Each step must have at least one `<premise>` and one `<conclusion>`
- You can have multiple `<premise>` tags within a single step
- The final conclusion of the entire reasoning chain should be in the last step's `<conclusion>`

## STEP SEPARATION RULES
- Step 1: `<step> ... </step>`
- [BLANK LINE]
- Step 2: `<step> ... </step>`
- [BLANK LINE]
- Step 3: `<step> ... </step>`
- etc.

## EXAMPLES

### Example 1: Simple Logic
**Input**: "All cats are animals. Fluffy is a cat. What is Fluffy?"
**Output**:
<step>
<premise>All cats are animals.</premise>
<premise>Fluffy is a cat.</premise>
<conclusion>Therefore, Fluffy is an animal.</conclusion>
</step>

\boxed{animal}

### Example 2: Mathematical Reasoning
**Input**: "What is 3 × 4? (A) 7 (B) 12 (C) 34"
**Output**:
<step>
<premise>Multiplication is repeated addition: 3 × 4 = 3 + 3 + 3 + 3</premise>
<conclusion>3 + 3 + 3 + 3 = 12</conclusion>
</step>

<step>
<premise>The calculation yields 12.</premise>
<premise>Option (B) is 12.</premise>
<conclusion>Therefore, the correct answer is option (B).</conclusion>
</step>

\boxed{B}

### Example 3: Conditional Reasoning
**Input**: "If it is raining, then I carry an umbrella. I am not carrying an umbrella. Is it raining?"
**Output**:
<step>
<premise>Conditional statement: If it is raining → I carry an umbrella</premise>
<conclusion>This establishes a logical relationship between rain and umbrella.</conclusion>
</step>

<step>
<premise>The contrapositive: If I am not carrying an umbrella → it is not raining</premise>
<conclusion>This is the logically equivalent form of the conditional.</conclusion>
</step>

<step>
<premise>I am not carrying an umbrella (given fact).</premise>
<premise>Applying the contrapositive: If not carrying umbrella → not raining</premise>
<conclusion>Therefore, it is not raining.</conclusion>
</step>

\boxed{No}

## COMMON MISTAKES TO AVOID
1. ❌ DON'T combine multiple inferences in one step
2. ❌ DON'T skip showing premises
3. ❌ DON'T forget to wrap premises and conclusions in tags
4. ❌ DON'T put reasoning after the \boxed{} answer
5. ❌ DON'T use markdown or other formatting inside the tags

## SPECIAL CASES
- If the problem has no choices, output only the reasoning steps
- If a step has multiple premises, list them in separate `<premise>` tags
- If you need to make an assumption, state it explicitly in a premise
- If the problem is mathematical, show calculations clearly

Now, solve the following problem.
""".strip()
    eval_path = "/home/linziyong/Desktop/Program/TreeRL/TreeRL/datasets/eval/logiqa.jsonl"
    num_gpus = 1

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    import torch
    
    def tokenize_fn(texts, max_length=2048, device="cpu", system_prompt=None):
        if isinstance(texts[0], (list, tuple)) and len(texts[0]) > 0:
            user_input = texts[0][0]
        else:
            user_input = texts[0]
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_input})
        
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        sample_input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
        
        if len(sample_input_ids) > max_length:
            sample_input_ids = sample_input_ids[-max_length:]
        
        output = {
            "input_ids": [torch.tensor(sample_input_ids)],
            "messages": messages
        }
        return output

    def decode_fn(ids):
        return tokenizer.decode(ids, skip_special_tokens=False)

    # Read input data
    with open(eval_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]

    # Split data across GPUs
    data_batches = [datas[i::num_gpus] for i in range(num_gpus)]

    # Create a process for each GPU
    processes = []
    # process_single_data_for_each_gpu(data_batches[0][:], 0, MODEL_PATH, evaluator_urls, extractor_urls, tokenize_fn, system_prompt=system_prompt_2)
    process_single_data_for_each_gpu_mcts(data_batches[0][:], 0, MODEL_PATH, tokenize_fn, decode_fn, EOS_TOKENS, system_prompt=system_prompt_2)
    
    # for gpu_id, data_batch in enumerate(data_batches):
    #     p = Process(target=process_single_data_for_each_gpu, args=(
    #         data_batch, gpu_id, MODEL_PATH, evaluator_urls, extractor_urls, eos_tokens, tokenize_fn))
    #     processes.append(p)
    #     p.start()

    # # Wait for all processes to complete
    # for p in processes:
    #     p.join()
