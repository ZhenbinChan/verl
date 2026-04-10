import random
from vllm import LLM
from transformers import AutoTokenizer
from entropy_chain_local_manager import EntropyGuidedChainLocalManager
from parallel_mcts_local_manager import ParallelMCTSLocalManager
from tree_node import pass_rate

import torch
import os
import json
import pandas as pd

from tqdm import tqdm
from json import JSONEncoder
from enum import Enum

class EnumJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return super().default(obj)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--eval_path", type=str, default="data/logiqa.jsonl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--generation_strategy", type=str, default="mcts", choices=["mcts", "treerl"])
    parser.add_argument("--evaluation_strategy", type=str, default="nli", choices=["token-entropy", "segment-entropy", "nli", "fol", "self-eval", "model-eval"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--use_all_terminals", action="store_true", help="Whether to use all terminal nodes for evaluation and selection, or only the ones with highest reward")
    parser.add_argument("--num_traces", type=int, default=30)
    # TreeRL parameters
    parser.add_argument("--m", type=int, default=1, help="TreeRL parameter m")
    parser.add_argument("--n", type=int, default=4, help="TreeRL parameter n")
    parser.add_argument("--l", type=int, default=3, help="TreeRL parameter l")
    parser.add_argument("--t", type=int, default=3, help="TreeRL parameter t")
    # MCTS parameters
    parser.add_argument("--random_pick", action="store_true", help="Whether to randomly pick nodes during selection")
    parser.add_argument("--max_children", type=int, default=4, help="Maximum number of children to expand for each node")
    parser.add_argument("--min_children", type=int, default=4, help="Minimum number of children to expand for each node when using shallow_enwide")
    parser.add_argument("--shallow_enwide", action="store_true", help="Whether to use the 'shallow and wide' MCTS variant") 
    parser.add_argument("--concurrent_num", type=int, default=4, help="Number of nodes to select for expansion in each iteration, also the number of parallel inferences")
    
    usr_args = parser.parse_args()

    tokenizer_path = usr_args.tokenizer_path
    eos_tokens = [151643]
    
    # system_prompt = 'please reason step by step with steps separated by "\n\n", and put the index of the correct answer within \\boxed{{}}.'
    prompt_dir = "mcts_utils/prompts"
    with open(os.path.join(prompt_dir, "Generation1.txt"), "r") as f:
        system_prompt = f.read()

    # Read input data
    with open(usr_args.eval_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]

    df = pd.DataFrame(datas)
    data_batch = [
        {
            "Question": batch["Question"].tolist(),
            "Answer": batch["Answer"].tolist()
        }
        for _, batch in df.groupby(df.index // usr_args.batch_size)
    ]

    use_api_generation = "gpt" in tokenizer_path.lower()
    if not use_api_generation:    
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
        import tiktoken
        tokenizer = tiktoken.encoding_for_model(tokenizer_path)
        llm = None

    
    def encode_fn(texts, max_length=2048, device="cpu", system_prompt=None):
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

    if usr_args.generation_strategy == "entropy_chain":
        args = {
            # TreeRL parameters
            "m": usr_args.m,
            "n": usr_args.n,
            "l": usr_args.l,
            "t": usr_args.t,
            "use_diverse_sampling" : False, # 选择TopK中最多样的轨迹进行迭代
            # Generating parameters
            "temperature": 1.0,
            "top_p": 0.9,
            "max_token_num" : 4096, # 每轮迭代生成的最大 token 数
            "training_type": "general",
            "system_prompt": system_prompt,
            "eos_tokens": eos_tokens,
            # Evaluating node parameters
            "use_pure_binary": True, # node.score = reward where reward = 1 / 0
            "use_pure_RM" : True, # node.score = sigmoid(a*(value-b)) if use_pure_RM else (reward + 0.5 * sigmoid(value))
            "a": 0.5,
            "b": 0.5,
            # TreeRL Selecting terminals parameters
            "inner_repetition_penalty" : False, # if leaf.finish_reason != "stop": leaf.R = -1
            "overall_norm_style": "token", # 归一化node.accumulated_value = node.accumulated_value - mean * node.terminal_in_subtree
            "use_all_terminals": usr_args.use_all_terminals, # 是否选择所有的轨迹
            "num_traces": usr_args.num_traces, # 最终选择轨迹数量
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
            "evaluation_strategy": usr_args.evaluation_strategy
        }
        manager = EntropyGuidedChainLocalManager(
            args=args,
            llm=llm,
            tokenizer=tokenizer,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            eos_tokens_set=eos_tokens,
        )
    else:
        args = {
            # MCTS parameters
            "max_depth": 40, # 最大树深
            "max_node_per_depth": 18, # 每层最大节点数
            "max_nodes": 256, # 最大节点数
            "max_children": usr_args.max_children, # 每个节点最大孩子数 
            "min_children": usr_args.min_children,  # 每个节点最少孩子数, 当使用shallow_enwide时
            "shallow_enwide": usr_args.shallow_enwide, # "Shallow and wide"（浅而宽）一种广度优先倾向的MCTS变体
            "concurrent_num": usr_args.concurrent_num, # 每次选择扩展的节点数量，也是每次并行推理的数量
            "first_token_temperature": 0, # 生成子节点时, 第一个 token 的 temperature, 用于增加多样性
            "exploration_constant": 0.5, # (node.value+offset+1e-8)/2 + self.exploration_constant * math.sqrt(math.log(node.parent.visits + 1) / (node.visits + self.epsilon))
            "backprop": True, # 是否进行奖励反向传播更新节点价值
            "look_ahead": 0,  # 无用参数
            "max_time_use": 360,
            "backbone": "qwen",
            "random_pick": usr_args.random_pick, # 每次选择扩展节点时，是否随机选择
            "pass_k": 30, # 超过pass_k的节点不再选择扩展
            # Generating parameters
            "temperature": 1.0,
            "top_p": 0.9,
            "max_token_num" : 4096, # 每轮迭代生成的最大 token 数
            "training_type": "general",
            "system_prompt": system_prompt,
            "eos_tokens": eos_tokens,
            # Evaluating node parameters
            "use_pure_binary": True,
            "use_pure_RM": False,
            "a": 0.5,
            "b": 0.5,
            # MCTS Selecting terminals parameters
            "use_all_terminals": usr_args.use_all_terminals, # 是否选择所有的轨迹
            "num_traces": usr_args.num_traces, # 最终选择的轨迹数量, 用于评测
            "select_correct_leaf": False, # 是否至少选择一个正确的叶子节点
            "use_weighted_value": False,
            # Gathering paths parameters
            "use_orm_reward": False,
            "use_chain_reward":  False,
            "step_level_norm": False,
            "use_state_value_reward": False,
            "use_value_only": False,
            "average_one_generation": False,
            "advantage_mix_allancestor": False,
            # Custom parameters
            "use_api_generation": use_api_generation,
            "enable_info": False,
            "check_step_validity": False,
            "evaluation_strategy": usr_args.evaluation_strategy
        }
        
        manager = ParallelMCTSLocalManager(
            args=args,
            llm=llm,
            tokenizer=tokenizer,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            eos_tokens_set=eos_tokens,
        )
    
    score_list = []
    selected_score_list = []
    for data in tqdm(data_batch, desc=f"Progress"):
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
    print(f"Finished processing. Scores: {avg_score:.4f}, avg_selected_score: {avg_selected_score:.4f}")

    results = {
        "args": args,
        "avg_score": avg_score,
        "avg_selected_score": avg_selected_score,
        "score_list": score_list,
        "selected_score_list": selected_score_list,
        "num_samples": len(score_list)
    }

    output_file = f"results/{tokenizer_path.split('/')[-1]}_{usr_args.generation_strategy}_{usr_args.evaluation_strategy}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=EnumJSONEncoder)
