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

import torch
import math
import os
import json
from multiprocessing import Process
from filelock import FileLock
import pandas as pd

from tqdm import tqdm
from json import JSONEncoder
from enum import Enum

class EnumJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return super().default(obj)


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
    data_batch, gpu_id, tokenizer_path, tokenize_fn, detokenize_fn, eos_tokens, system_prompt=None
):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = {
        # TreeRL parameters
        "m": 1,
        "n": 4,
        "l": 3,
        "t": 3,
        "training_type": "general",
        "system_prompt": system_prompt,
        "generate_max_len" : 4096, # 每轮迭代生成的最大 token 数
        "use_diverse_sampling" : False, # 选择TopK中最多样的轨迹进行迭代
        # Generating parameters
        "temperature": 1.0,
        "top_p": 0.9,
        "eos_tokens": eos_tokens,
        # Evaluating node parameters
        "use_pure_binary": True, # node.score = reward where reward = 1 / 0
        "use_pure_RM" : True, # node.score = sigmoid(a*(value-b)) if use_pure_RM else (reward + 0.5 * sigmoid(value))
        "a": 0.5,
        "b": 0.5,
        # Selecting terminals parameters
        "inner_repetition_penalty" : False, # if leaf.finish_reason != "stop": leaf.R = -1
        "overall_norm_style": "token", # 归一化node.accumulated_value = node.accumulated_value - mean * node.terminal_in_subtree
        "use_all_terminals": False, # 是否选择所有的轨迹
        "num_traces": 12, # 最终选择轨迹数量
        "balance_ratio": -0.2, # 最终选择的轨迹中，正确轨迹的比例，负数表示使用启发式选择节点（用于评测）
        "use_weighted_value": False, # 根据所选轨迹更改node.accumulated_value
        "weighted_value_style": "sqrt", # 更改node.accumulated_value的方式
        # Gathering paths parameters
        "use_orm_reward" : False, # 更改node.value的各种方式
        "use_chain_reward" : False,
        "step_level_norm" : False,
        "use_state_value_reward" : False,
        "use_value_only" : True,
        "average_one_generation" : False,
        "advantage_mix_allancestor" : False,
        # Custom parameters
        "use_api_generation": "gpt" in tokenizer_path.lower(),
        "enable_info": False,
        "check_step_validity": False,
        "evaluation_strategy": 'segment-entropy' # 'token-entropy', 'segment-entropy', 'nli', 'fol', 'model-evaluation'
    }
    if not args["use_api_generation"]:    
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        llm = LLM(
            model=tokenizer_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            seed=3407,
        )
    else:
        tokenizer = tiktoken.encoding_for_model(tokenizer_path)
        llm = None

    manager = EntropyGuidedChainLocalManager(
        args=args,
        llm=llm,
        tokenizer=tokenizer,
        encode_fn=tokenize_fn,
        decode_fn=detokenize_fn,
        eos_tokens_set=eos_tokens,
    )

    df = pd.DataFrame(data_batch)
    data_batch = [
        {
            "Question": batch["Question"].tolist(),
            "Answer": batch["Answer"].tolist()
        }
        for _, batch in df.groupby(df.index // 1)
    ]

    score_list = []
    selected_score_list = []
    for data in tqdm(data_batch, desc=f"GPU {gpu_id} progress"):
        item = {
            "problem": data["Question"],
            "golden_answer": data["Answer"],
        }
        result = manager.process_single_item(item, args)
        score = result['raw_avg_reward']
        selected_score = [pass_rate(paths) for paths in result['paths']]

        score_list.extend(score)
        selected_score_list.extend(selected_score)
        avg_score = sum(score_list) / len(score_list)
        avg_selected_score = sum(selected_score_list) / len(selected_score_list)
        print(f"avg_score: {avg_score:.4f}, avg_selected_score: {avg_selected_score:.4f}")
    print(f"GPU {gpu_id} finished processing. Scores: {avg_score:.4f}")

    avg_score = sum(score_list) / len(score_list)
    avg_selected_score = sum(selected_score_list) / len(selected_score_list)
    results = {
        "args": args,
        "avg_score": avg_score,
        "avg_selected_score": avg_selected_score,
        "score_list": score_list,
        "selected_score_list": selected_score_list,
        "num_samples": len(score_list)
    }

    output_file = f"results/{'GPT4o-mini' if args['use_api_generation'] else tokenizer_path.split('/')[-1]}_tau_{args['temperature']}_mnlt_{args['m']}_{args['n']}_{args['l']}_{args['t']}_{args['evaluation_strategy']}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=EnumJSONEncoder)


def process_single_data_for_each_gpu_mcts(
    data_batch, gpu_id, tokenizer_path, tokenize_fn, detokenize_fn, eos_tokens, system_prompt=None
):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = {
        # MCTS parameters
        "max_depth": 40, # 最大树深
        "max_node_per_depth": 18, # 每层最大节点数
        "max_nodes": 256, # 最大节点数
        "max_children": 4, # 每个节点最大孩子数 
        "min_children": 4,  # 每个节点最少孩子数, 当使用shallow_enwide时
        "shallow_enwide": False, # "Shallow and wide"（浅而宽）一种广度优先倾向的MCTS变体
        "prompt_key": "problem", 
        "answer_key": "golden_answer", 
        "backbone": "qwen",
        "concurrent_num": 4, # 每次选择扩展的节点数量，也是每次并行推理的数量
        "first_token_temperature": 0, # 生成子节点时, 第一个 token 的 temperature, 用于增加多样性
        "exploration_constant": 0.5, # (node.value+offset+1e-8)/2 + self.exploration_constant * math.sqrt(math.log(node.parent.visits + 1) / (node.visits + self.epsilon))
        "backprop": True, # 是否进行奖励反向传播更新节点价值
        "look_ahead": 0,  # 无用参数
        "prompt_max_len": 4096,
        "max_token_num": 4096,
        "max_time_use": 360, 
        "random_pick": True, # 每次选择扩展节点时，是否随机选择
        # Generating parameters
        "temperature": 1.0,
        "top_p": 0.9,
        "eos_tokens": eos_tokens,
        # Evaluating node parameters
        "use_pure_binary": True,
        "use_pure_RM": False,
        "a": 0.5,
        "b": 0.5,
        # Selecting terminals parameters
        "path_num": 30, # 最终选择的轨迹数量, 用于评测
        "pass_k": 30, # 无用参数，功能同 path_num 
        "select_correct_leaf": False, # 是否至少选择一个正确的叶子节点
        "use_weighted_value": False,
        "use_all_terminals": True,
        # Gathering paths parameters
        "use_orm_reward": False,
        "use_chain_reward":  False,
        "step_level_norm": False,
        "use_state_value_reward": False,
        "use_value_only": False,
        "average_one_generation": False,
        "advantage_mix_allancestor": False,
        # Custom parameters
        "use_api_generation": "gpt" in tokenizer_path.lower(),
        "enable_info": False,
        "check_step_validity": False,
        "evaluation_strategy": EvaluationStrategy.MODEL_EVALUATION
    }
    if not args["use_api_generation"]:    
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        llm = LLM(
            model=tokenizer_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.75,
            seed=3407,
        )
    else:
        tokenizer = tiktoken.encoding_for_model(tokenizer_path)
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
            evaluation_strategy=args["evaluation_strategy"],
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
    MODEL_PATH = "/home/linziyong/Desktop/Model/Qwen2.5-3B-Instruct" # "gpt-4o-mini-2024-07-18" # 
    EOS_TOKENS = [151643]
    # SYSTEM_PROMPT = 'please reason step by step with steps separated by "\n\n", and put the index of the correct answer within \\boxed{{}}.'
    prompt_dir = "mcts_utils/prompts"
    with open(os.path.join(prompt_dir, "Generation1.txt"), "r") as f:
        SYSTEM_PROMPT = f.read()

    eval_path = "data/logiqa.jsonl"
    num_gpus = 1

    use_api_generation = "gpt" in MODEL_PATH.lower()
    if not use_api_generation:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)
    else:
        import tiktoken
        tokenizer = tiktoken.encoding_for_model(MODEL_PATH)

    
    def tokenize_fn(texts, max_length=2048, device="cpu", system_prompt=None):
        if isinstance(texts[0], (list, tuple)) and len(texts[0]) > 0:
            user_input = texts[0][0]
        else:
            user_input = texts[0]
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_input})
        
        if not use_api_generation:
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            sample_input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
            
            if len(sample_input_ids) > max_length:
                sample_input_ids = sample_input_ids[-max_length:]
            
            sample_input_ids = [torch.tensor(sample_input_ids)]
        else:
            sample_input_ids = None
        
        output = {
            "input_ids": sample_input_ids,
            "messages": messages
        }
        return output

    def decode_fn(ids):
        if not use_api_generation:
            return tokenizer.decode(ids, skip_special_tokens=False)
        return tokenizer.decode(ids)

    # Read input data
    with open(eval_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]

    # Split data across GPUs
    data_batches = [datas[i::num_gpus] for i in range(num_gpus)]

    # Create a process for each GPU
    processes = []
    process_single_data_for_each_gpu(data_batches[0][:], 0, MODEL_PATH, tokenize_fn, decode_fn, EOS_TOKENS, SYSTEM_PROMPT)
    # process_single_data_for_each_gpu_mcts(data_batches[0][:], 0, MODEL_PATH, tokenize_fn, decode_fn, EOS_TOKENS, system_prompt=SYSTEM_PROMPT)
    
    # for gpu_id, data_batch in enumerate(data_batches):
    #     p = Process(target=process_single_data_for_each_gpu, args=(
    #         data_batch, gpu_id, MODEL_PATH, evaluator_urls, extractor_urls, eos_tokens, tokenize_fn))
    #     processes.append(p)
    #     p.start()

    # # Wait for all processes to complete
    # for p in processes:
    #     p.join()
