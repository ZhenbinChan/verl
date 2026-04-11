from __future__ import annotations
from typing import Optional,Callable, Dict, Any, List
import random
import math
from collections import deque
import time
from typing import List
from pydantic import BaseModel
from evaluation import (
    check_result,
    generate_logits,
    test_sglang_model,
    test_glm_model,
    get_qwen_remote_reward_model_value,
    query_local_vllm_completions_ids,
    QWEN_SYSTEM_PROMPT,
    UNDERSTANDING_PROMPT,
    QWEN_QA_PROMPT,
    GLM_QA_PROMPT,
    top_k_sampling,
    self_reward_generation,
    check_step_format,
    model_reward_generation,
    query_openai_api_completions_ids_slow,
    query_openai_api_completions_ids_fast,
)
from tree_node import (
    pass_rate, 
    gather_paths, 
    print_tree
)
import numpy as np
from utils import *

ROOT_UCT_SCORE = 10_000
QUEUE_SIZE = 10000
NUM_PROCESS = 50

from collections import Counter

def get_stops(backbone="glm"):
    if backbone == "glm":
        return [271, 151336, 151329,151338, 2533, 382, 1447, 21467, 692]
    elif backbone == "qwen":
        return [151645, 271, 2533, 382, 1447, 21518, 692, 1339]
    
def find_repeated_patterns(s, pattern_length=50, threshold=20):
    from collections import defaultdict
    # pattern_counts = defaultdict(int)
    # for i in range(len(s) - pattern_length + 1):
    #     pattern = s[i:i + pattern_length]
    #     pattern_counts[pattern] += 1
    # repeated_patterns = {pattern: count for pattern, count in pattern_counts.items() if count >= threshold}
    # 生成 N-grams
    ngrams = [s[i:i + pattern_length]
              for i in range(len(s) - pattern_length + 1)]
    # 统计 N-gram 出现次数
    ngram_counts = Counter(ngrams)
    # 筛选出重复的 N-gram
    repeated_patterns = {gram: count for gram,
                         count in ngram_counts.items() if count > threshold}
    return repeated_patterns

class MCTSNode(BaseModel):
    state: List[int]
    answer: str = ""
    answer_token : List[int] = []
    aggregate_answer: str = ""
    aggregate_answer_token: List[int] = []
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    visits: int = 0
    R : float = 0
    value : float = 0
    reward_samples: list[int] = []
    depth: int = 0
    main_chain: bool = False
    terminal: bool = False
    max_children: int = 3
    terminal_in_subtree: int = 0
    correct_terminal_in_subtree: int = 0
    selected_terminal_in_subtree: int = 0
    accumulated_value: float = 0
    visited_terminal: int = 0
    repeat:bool = False
    tree_idx: int = 0
    node_idx: int = 0
    is_correct: Optional[bool] = None

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, value={self.value:.2f}, visits={self.visits})"

    def __eq__(self, other):
        if isinstance(other, MCTSNode):
            # return self.state == other.state and self.answer == other.answer
            return self.node_idx == other.node_idx and self.tree_idx == other.tree_idx
        return False

    def __hash__(self):
        return hash((self.node_idx))

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)


class ParallelMCTSLocalManager:
    def __init__(
        self,
        args: Dict[str, Any],
        llm: List[Any],
        tokenizer: Any,
        encode_fn: Callable,
        decode_fn: Callable,
        eos_tokens_set: List[int] = None
    ):
        """
        初始化MCTS管理器
        
        Args:
            args: 参数字典
            llms: LLM模型列表
            tokenizer: 分词器
            encode_fn: 编码函数
            decode_fn: 解码函数
            eos_tokens_set: 结束token集合
        """
        self.args = args
        self.llm = llm
        self.tokenizer = tokenizer
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.eos_tokens_set = eos_tokens_set
        self.system_prompt = args.get("system_prompt", None)
        
        # 从参数中提取配置
        self.max_nodes = args["max_nodes"]
        self.max_token_num = args["max_token_num"]
        self.max_time_use = args["max_time_use"]
        self.max_depth = args.get("max_depth", 40)
        self.max_node_per_depth = args.get("max_node_per_depth", 16)
        self.max_children = args.get("max_children", 3)
        self.min_children = args.get("min_children", 2)
        self.shallow_enwide = args.get("shallow_enwide", False)
        self.concurrent_num = args.get("concurrent_num", 4)

        self.exploration_constant = args.get("exploration_constant", 1.0)
        self.epsilon = args.get("epsilon", 1e-10)

        self.pass_k = args.get("pass_k", 4)
        self.backbone = args.get("backbone", "qwen") # NOTE
        self.temperature = args.get("temperature", 0.9)
        self.top_p = args.get("top_p", 0.9)
        
        self.backprop = args.get("backprop", True)
        self.selection_policy = args.get("selection_policy", "importance_sampling")
        self.first_token_temperature = args.get("first_token_temperature", False)
        self.step_level_norm = args.get("step_level_norm", True)
        self.random_pick = args.get("random_pick", False)
        self.use_weighted_value = args.get("use_weighted_value", False)
        self.use_orm_reward = args.get("use_orm_reward", False)
        self.select_correct_leaf = args.get("select_correct_leaf", False)
        self.use_chain_reward = args.get("use_chain_reward", False)
        self.use_state_value_reward = args.get("use_state_value_reward", False)
        self.use_pure_RM = args.get("use_pure_RM", False)
        self.use_pure_binary = args.get("use_pure_binary", False)
        self.average_one_generation = args.get("average_one_generation", False)
        self.use_value_only = args.get("use_value_only", False)
        self.a = args.get("a", 0.5)
        self.b = args.get("b", -2.898)

        self.use_api_generation = args.get("use_api_generation", False)
        self.enable_info = args.get("enable_info", False)
        self.evaluation_strategy = args.get("evaluation_strategy", "self_evaluation")
        self.check_step_validity = args.get("check_step_validity", False)
        

    def info(self, message: str, enable_info: Optional[bool] = None):
        enable_info = self.enable_info if enable_info is None else enable_info
        if enable_info:
            print(f"{message}")
    
    def self_evaluate(self, node: MCTSNode, is_terminated: bool, problem: str, golden_answer) -> float:
        label2score = {
            "ENTAILMENT": 1,
            "NEUTRAL": 0,
            "CONTRADICTION": -1,
            "UNKNOWN": 0,
            "ERROR": 0
        }
        
        if is_terminated:
            if node.repeat:
                reward = 0
                node.is_correct = False
            else:
                answer, correct = check_result(problem, node.answer, golden_answer)
                node.is_correct = correct
                
                if answer:
                    if self.evaluation_strategy == "self-eval":
                        reward = self_reward_generation(
                            self.llm, problem, node.aggregate_answer, 
                            self.encode_fn, is_terminated
                        )
                    elif self.evaluation_strategy == "model-eval":
                        reward = model_reward_generation(
                            problem, node.aggregate_answer, is_terminal=is_terminated
                        )
                    elif self.evaluation_strategy == "fol":
                        raise NotImplementedError("FOL evaluation is not implemented in self_evaluate method")
                        # reward = get_fol_reward(problem, node.aggregate_answer)
                    elif self.evaluation_strategy == "nli":
                        parsed_chain = parse_reasoning_steps(node.aggregate_answer)
                        results_nli = verify_steps_nli(parsed_chain)
                        segment_scores = [label2score[result['label']] * result['score'] for result in results_nli]
                        reward = sum(segment_scores) / len(segment_scores) if segment_scores else 0
                    else:
                        reward = float(random.randint(0, 1))
                else:
                    reward = 0
                    
                if self.use_pure_binary:
                    node.accumulated_value = reward
                else:
                    value = get_qwen_remote_reward_model_value(
                        urls=RM_URLS,  # 需要从外部导入
                        question=problem,
                        response=node.aggregate_answer
                    )
                    if self.use_pure_RM:
                        x = self.a * (value - self.b)
                        result = 1 / (1 + math.exp(-x))
                        node.accumulated_value = result
                    else:
                        sigmoid_value = 1 / (1 + math.exp(-value))
                        coeff = 0.5
                        value = reward + coeff * sigmoid_value
                        node.accumulated_value = value
        else:
            if self.check_step_validity:
                is_valid_step, _, errors = check_step_format(node.answer)
            else:
                is_valid_step = True
                
            if is_valid_step:
                if self.evaluation_strategy == "self-eval":
                    reward = self_reward_generation(
                        self.llm, problem, node.aggregate_answer, 
                        self.encode_fn, is_terminated
                    )
                elif self.evaluation_strategy == "model-eval":
                    reward = model_reward_generation(
                        problem, node.aggregate_answer, is_terminal=is_terminated
                    )
                elif self.evaluation_strategy == "fol":
                    raise NotImplementedError("FOL evaluation is not implemented in self_evaluate method")
                    # reward = get_fol_reward(problem, node.aggregate_answer)
                elif self.evaluation_strategy == 'nli':
                    parsed_chain = parse_reasoning_steps(node.aggregate_answer)
                    results_nli = verify_steps_nli(parsed_chain)
                    segment_scores = [label2score[result['label']] * result['score'] for result in results_nli]
                    reward = sum(segment_scores) / len(segment_scores) if segment_scores else 0
                else:
                    reward = float(random.randint(0, 1))
            else:
                reward = 0
                
                
        return reward

    def backpropagate(self, node: MCTSNode, gamma: float = 0.9, main_chain: bool = False):
        self.info("backpropagate")
        parent = node.parent
        node.visits += 1
        
        if main_chain:
            node.main_chain = True
            
        if node.children:
            nume, deno = 0, 0
            for child in node.children:
                reward = child.R - node.R
                q_value = reward + gamma * child.value
                nume += q_value * child.visits
                deno += child.visits
                
            if deno:
                node.value = nume / deno
            else:
                self.info(f"Fail to process value, nume: {nume}, deno: {deno}")
        else:
            node.value = node.R
            
        if parent:
            self.backpropagate(parent, gamma, main_chain)

    def leaf_backpropagate(self, node: MCTSNode):
        if node.terminal and node.main_chain:
            node.terminal_in_subtree += 1
            node.correct_terminal_in_subtree += 1
            parent = node.parent
            while parent:
                parent.terminal_in_subtree += 1
                parent.correct_terminal_in_subtree += 1
                parent.accumulated_value += node.accumulated_value
                parent = parent.parent
        elif node.terminal:
            node.terminal_in_subtree += 1
            parent = node.parent
            while parent:
                parent.terminal_in_subtree += 1
                parent.accumulated_value += node.accumulated_value
                parent = parent.parent

    def selected_backpropagate(self, node: MCTSNode):
        node.selected_terminal_in_subtree += 1
        parent = node.parent
        while parent:
            parent.selected_terminal_in_subtree += 1
            parent = parent.parent

    def uct(self, node: MCTSNode, offset: int = 1) -> float:
        if not node.parent:
            return ROOT_UCT_SCORE
            
        return (node.value + offset + 1e-8) / 2 \
            + self.exploration_constant * math.sqrt(
                math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
            )

    def _is_fully_expanded(self, node: MCTSNode, depth_node_count: Dict[int, int]) -> bool:
        next_depth = node.depth + 1
        next_depth_count = depth_node_count.get(next_depth, 0)
        if next_depth_count >= self.max_node_per_depth:
            return True
        return len(node.children) >= node.max_children or node.terminal

    def _weighted_sample_no_replacement(self, candidates: List, weights: List[float], k: int) -> List[int]:
        total_candidates = len(candidates)
        k = min(total_candidates, k)
        
        if k >= total_candidates:
            return list(range(total_candidates))
        if k <= 0 or total_candidates == 0:
            return []
            
        indices = list(range(total_candidates))
        selected_indices = []
        
        for _ in range(k):
            try:
                chosen_index = random.choices(indices, weights=weights, k=1)[0]
            except ValueError:
                uct_scores = [self.uct(node, 2) for node in candidates]
                chosen_index = random.choices(indices, weights=uct_scores, k=1)[0]
                
            selected_indices.append(chosen_index)
            remove_index = indices.index(chosen_index)
            indices.pop(remove_index)
            weights.pop(remove_index)
            candidates.pop(remove_index)
                
        return selected_indices

    def select_node(self, root: MCTSNode, depth_node_count: Dict[int, int], k: int = 1, random_pick: bool = False) -> List[MCTSNode]:
        candidates = []
        to_consider = deque([root])
        
        while to_consider:
            current_node = to_consider.popleft()
            if not self._is_fully_expanded(current_node, depth_node_count):
                candidates.append(current_node)
            to_consider.extend(current_node.children)
            
        if not candidates:
            return []
            
        if random_pick:
            return random.sample(candidates, min(k, len(candidates)))
            
        if self.selection_policy == "greedy":
            return sorted(candidates, key=self.uct, reverse=True)[:k]
            
        elif self.selection_policy == "importance_sampling":
            uct_scores = [self.uct(node) for node in candidates]
            candidates_copy = candidates.copy()
            selected_indices = self._weighted_sample_no_replacement(
                candidates_copy, uct_scores, k
            )
            return [candidates[i] for i in selected_indices]
            
        elif self.selection_policy == "pairwise_comparison":
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [
                (i, j) for i in range(len(candidates)) 
                for j in range(len(candidates))
            ]
            pair_weights = [
                max(uct_scores[i], uct_scores[j]) - min(uct_scores[i], uct_scores[j])
                for i, j in pairs
            ]
            
            selected_pair_indices = random.choices(
                range(len(pairs)), weights=pair_weights, k=min(k, len(pair_weights))
            )
            
            selected_nodes = []
            for pair_idx in selected_pair_indices:
                selected_candidate_idx = max(
                    pairs[pair_idx], key=lambda x: uct_scores[x]
                )
                if candidates[selected_candidate_idx] not in selected_nodes:
                    selected_nodes.append(candidates[selected_candidate_idx])
                if len(selected_nodes) >= k:
                    break
            return selected_nodes
            
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    def process_single_item(self, item: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        problems = item["problem"]
        answers = item["golden_answer"]
        system_prompt = args.get("system_prompt", None)
            
        if args["training_type"] == "general":
            paths, raw_avg_reward = self.parallel_mcts(problems, answers, system_prompt=system_prompt, args=args)
        else:
            paths = self.parallel_mcts(problems, answers, system_prompt=system_prompt, args=args)
        
        results = {
            "problem": problems,
            "golden_answer": answers,
            "paths": paths,
            "raw_avg_reward": raw_avg_reward if args["training_type"] == "general" else None
        }
        return results

    def parallel_mcts(
        self,
        problems: List[str],
        answers: List[str],
        system_prompt=None,
        args: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        batch_size = len(problems)

        # 初始化统计信息
        root_lst = []
        node_number_lst = [1] * batch_size
        leaf_num_lst = [0] * batch_size
        leaves_lst = [[] for _ in range(batch_size)]
        depth_node_count_lst = [{0: 1} for _ in range(batch_size)]

        for tree_idx, problem_str in enumerate(problems):
            encode_output = self.encode_fn([[problem_str]], 4096, device="cpu", system_prompt=system_prompt)
            init_state = encode_output["messages"] if self.use_api_generation else encode_output["input_ids"][0].tolist()
            root_lst.append(MCTSNode(state=init_state, max_children=self.max_children, tree_idx=tree_idx))

        start_time = time.time()
        
        while (max(node_number_lst) < self.max_nodes and time.time() - start_time < self.max_time_use):
            selected_nodes = []

            for root, depth_node_count in zip(root_lst, depth_node_count_lst):
                if node_number_lst[root.tree_idx] >= self.max_nodes:
                    continue
                selected_nodes.extend(
                    self.select_node(root, depth_node_count, k=self.concurrent_num, random_pick=self.random_pick)
                )
            
            if not selected_nodes:
                self.info("Terminated because no node to expand")
                break

            # 扩展节点
            expand_results, node_number_lst = self.expand(
                selected_nodes, problems, answers, node_number_lst
            )
            
            # 处理扩展结果
            for node, child_num in expand_results["node_results"]:
                if child_num == 0:
                    self.info(f"Cannot expand node {node}")
                    continue
                    
                tree_idx = node.tree_idx
                depth_node_count_lst[tree_idx][node.depth + 1] = (
                    depth_node_count_lst[tree_idx].get(node.depth + 1, 0) + child_num
                )
                
                child_terminal_exists = False
                for child in node.children:
                    is_terminated = child.terminal
                    
                    if is_terminated and child.is_correct:
                        child.main_chain = True
                        if self.backprop:
                            self.backpropagate(child, main_chain=True)
                    else:
                        if self.backprop:
                            self.backpropagate(child)
                            
                    if is_terminated:
                        leaves_lst[tree_idx].append(child)

                        if not child_terminal_exists:
                            leaf_num_lst[tree_idx] += 1
                            child_terminal_exists = True

            if min(leaf_num_lst) >= self.pass_k:
                self.info(f"Terminated because reached {leaf_num_lst} leaf nodes")
                break

        # 后处理
        path_lst = []
        for root, leaves in zip(root_lst, leaves_lst):
            self.leaf_normalize(leaves)
            for leaf in leaves:
                self.leaf_backpropagate(leaf)
                
            if self.average_one_generation:
                self.update_accumulated_values(root)
                
            selected_terminals = self.select_terminal(
                leaves,
                args['num_traces'],
            )
            
            if self.use_weighted_value:
                for leaf in selected_terminals:
                    self.selected_backpropagate(leaf)
                self.weighted_update(root)

            paths = gather_paths(
                root,
                selected_terminals,
                args['num_traces'],
                use_orm_reward=args["use_orm_reward"],
                use_chain_reward=args["use_chain_reward"],
                step_level_norm=args["step_level_norm"],
                use_state_value_reward=args["use_state_value_reward"],
                use_value_only=args["use_value_only"],
                average_one_generation=args["average_one_generation"],
                advantage_mix_allancestor=args["advantage_mix_allancestor"]
            )
            self.info(f"Gathered {len(paths)} paths.")
            path_lst.append(paths)

        if args["training_type"] == "general":
            return path_lst, [root.correct_terminal_in_subtree / root.terminal_in_subtree for root in root_lst]
        return path_lst

    def expand(
        self, 
        nodes: List[MCTSNode], 
        problems: List[str],
        answers: List[str],
        leaf_num_count_lst: List[int]
    ) -> Dict[str, Any]:
        """扩展选中的节点"""
        if not nodes:
            return {"node_results": [], "total_tokens": 0}
            
        stops = get_stops(self.backbone)
        max_tokens_per_step = self.max_token_num
        max_attempts = 3
        children_map = {node: [] for node in nodes}
        
        attempts = 0
        while attempts < max_attempts:
            prompts, node_prompts_map = [], []
            
            for node in nodes:
                num_current_children = len(children_map[node])
                prompts.extend([node.state] * (node.max_children - num_current_children))
                node_prompts_map.extend([node] * (node.max_children - num_current_children))
                
            if not prompts:
                break
                
            attempts += 1
            next_tokens = []
            next_strs = []
            
            # 第一token温度采样
            if (self.first_token_temperature and random.random() < 0.5 and 
                not self.use_api_generation):
                self.info("Using first token temperature")
                first_tokens = top_k_sampling(
                    llm=self.llm,
                    prompts=prompts,
                    top_p=self.top_p,
                    skip_special_tokens=False,
                    stops=stops,
                )
                next_tokens = [random.choice(used_logprobs) for used_logprobs in first_tokens]
                next_strs = [self.decode_fn([next_token]) for next_token in next_tokens]
                prompts = [prompt + [next_token] for prompt, next_token in zip(prompts, next_tokens)]
                
            # 生成响应
            start_time = time.time()
            if not self.use_api_generation:
                responses_token, responses_str, finish_reasons, stop_tokens, token_nums = (
                    query_local_vllm_completions_ids(
                        prompts,
                        llm=self.llm,
                        n=1,
                        skip_special_tokens=True,
                        max_tokens=max_tokens_per_step,
                        stops=stops,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        model=self.backbone,
                    )
                )
            else:
                messages = [[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": self.decode_fn(prompt)},
                    {"role": "user", "content": "Please continue."}
                ] for prompt in prompts]
                
                responses_token, responses_str, finish_reasons, stop_tokens, token_nums = (
                    query_openai_api_completions_ids_fast(
                        messages,
                        n=1,
                        max_tokens=max_tokens_per_step,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        stops=['\n\n'],
                        tokenizer=self.tokenizer
                    )
                )
                
            end_time = time.time()
            self.info(f"Generation latency: {end_time - start_time:.2f} seconds")
            
            # 处理生成的响应
            for idx, (response_token_list, response_str_list, finish_reason_list, 
                     stop_token_list, token_num_list) in enumerate(
                zip(responses_token, responses_str, finish_reasons, stop_tokens, token_nums)
            ):
                node = node_prompts_map[idx]
                response_token = response_token_list[0]
                response = response_str_list[0]
                finish_reason = finish_reason_list[0]
                stop_token = stop_token_list[0]
                token_num = token_num_list[0]
                
                if next_tokens:
                    response = next_strs[idx] + response
                    response_token = [next_tokens[idx]] + response_token
                    
                if response is None:
                    continue
                    
                if not ((stop_token is None) or (stop_token in self.eos_tokens_set)):
                    stop_token_str = self.decode_fn([stop_token])
                    if not response.endswith(stop_token_str):
                        response += stop_token_str
                    if response_token[-1] != stop_token:
                        response_token += [stop_token]

                new_aggregate_answer = node.aggregate_answer + response
                new_aggregate_answer_token = node.aggregate_answer_token + response_token
                
                repeat = (len(new_aggregate_answer_token) > self.max_token_num) \
                    or (find_repeated_patterns(new_aggregate_answer) != {})
                
                finished = (stop_token is None) or (stop_token in self.eos_tokens_set) \
                    or (node.depth > self.max_depth) or repeat
                
                max_children = max(node.max_children / 2, self.min_children) if self.shallow_enwide else node.max_children
                    
                tree_idx = node.tree_idx

                child_node = MCTSNode(
                    state=node.state + response_token,
                    answer=response,
                    answer_token=response_token,
                    aggregate_answer=new_aggregate_answer,
                    aggregate_answer_token=new_aggregate_answer_token,
                    parent=node,
                    depth=node.depth + 1,
                    terminal=finished,
                    max_children=max_children,
                    repeat=repeat,
                    tree_idx=tree_idx,
                    node_idx=leaf_num_count_lst[tree_idx]
                )
                
                leaf_num_count_lst[tree_idx] += 1
                children_map[node].append(child_node)
                
            if all(len(children_map[node]) >= node.max_children for node in nodes):
                break
                
        # 评估子节点
        node_results = []
        for node, childrens in children_map.items():
            tree_idx = node.tree_idx
            problem = problems[tree_idx]
            answer = answers[tree_idx]

            if self.random_pick:
                for child in childrens:
                    if child.terminal:
                        child.R = self.self_evaluate(child, True, problem, answer)
                        child.value = child.R
            else:
                non_leaf_rewards = []
                non_leaf_indexes = []
                
                for i, child in enumerate(childrens):
                    if child.terminal:
                        child.R = self.self_evaluate(child, True, problem, answer)
                        child.value = child.R
                    else:
                        reward = self.self_evaluate(child, False, problem, answer)
                        non_leaf_rewards.append(reward)
                        non_leaf_indexes.append(i)
                        
                if non_leaf_rewards:
                    non_leaf_rewards = np.array(non_leaf_rewards)
                    normalized_rewards = (non_leaf_rewards - np.mean(non_leaf_rewards)) / (np.std(non_leaf_rewards) + 1e-8)
                    
                    for i, reward in zip(non_leaf_indexes, normalized_rewards):
                        childrens[i].R = reward
                        childrens[i].value = reward
                        
            node.max_children = len(childrens)
            node.children = childrens
            node_results.append((node, len(node.children)))
            
        return {
            "node_results": node_results,
        }, leaf_num_count_lst


    def judge_finished(self, is_stopped: bool, depth: int) -> bool:
        return is_stopped or depth > self.max_depth
            
    def leaf_normalize(self, nodes: List[MCTSNode]):
        leaf_correctness = [leaf.accumulated_value for leaf in nodes]
        _sum = sum(leaf_correctness)
        num = len(leaf_correctness) - 1
        
        if num == 0:
            return
            
        mean = [(_sum - leaf_correctness[i]) / num for i in range(len(leaf_correctness))]
        for i, leaf in enumerate(nodes):
            leaf.accumulated_value = leaf.accumulated_value - mean[i]

    def compute_accumulated_value(self, node: MCTSNode) -> float:
        if not node.children:
            return node.accumulated_value
            
        total_value = 0
        terminal_children = 0
        
        for child in node.children:
            if child.terminal_in_subtree > 0:
                terminal_children += 1
                total_value += self.compute_accumulated_value(child)
                
        node.accumulated_value = total_value / terminal_children if terminal_children > 0 else 0
        return node.accumulated_value

    def update_accumulated_values(self, root: MCTSNode):
        """更新累积值"""
        self.compute_accumulated_value(root)

    def compute_weighted_update(self, node: MCTSNode):
        """计算加权更新"""
        if node.selected_terminal_in_subtree > 0:
            node.accumulated_value = (
                node.accumulated_value * node.terminal_in_subtree / 
                node.selected_terminal_in_subtree
            )
        for child in node.children:
            self.compute_weighted_update(child)

    def weighted_update(self, root: MCTSNode):
        self.compute_weighted_update(root)

    def select_terminal(self, leaves: List[MCTSNode], num_traces: int) -> bool:
        """选择终止节点"""
        parent_to_children = {}
        
        if len(leaves) < 3:
            return leaves
            
        correct_leaf_parent = None
        correct_leaf = None
        
        for leaf in leaves:
            if leaf.main_chain:
                correct_leaf_parent = leaf.parent
                correct_leaf = leaf
            parent = leaf.parent
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(leaf)
            
        total_sum = len(parent_to_children)
        
        if correct_leaf_parent is not None:
            assert correct_leaf is not None, "Correct leaf is None"
            self.info("Got correct leaf!")
            
        if not self.select_correct_leaf:
            self.info("Do not manually select correct leaf")
            correct_leaf = None
            correct_leaf_parent = None
            
        if total_sum == num_traces:
            selected_terminals = []
            for parent, children in parent_to_children.items():
                if parent == correct_leaf_parent and correct_leaf is not None:
                    selected_terminals.append(correct_leaf)
                else:
                    selected_terminals.append(random.choice(children))
            
        elif total_sum > num_traces:
            if correct_leaf is None:
                selected_parents = random.sample(list(parent_to_children.keys()), num_traces)
                selected_terminals = []
                for parent in selected_parents:
                    selected_terminals.append(random.choice(parent_to_children[parent]))
            else:
                other_parents = [p for p in parent_to_children.keys() if p != correct_leaf_parent]
                selected_parents = random.sample(other_parents, num_traces - 1)
                selected_terminals = [correct_leaf]
                for parent in selected_parents:
                    selected_terminals.append(random.choice(parent_to_children[parent]))
        else:
            if correct_leaf is None:
                selected_terminals = []
                for parent, children in parent_to_children.items():
                    selected_terminals.append(random.choice(children))
                    
                if len(selected_terminals) < num_traces:
                    k = 0
                    while len(selected_terminals) < num_traces:
                        added_in_this_round = False
                        for parent, children in parent_to_children.items():
                            if k < len(children) and children[k] not in selected_terminals:
                                selected_terminals.append(children[k])
                                added_in_this_round = True
                                if len(selected_terminals) >= num_traces:
                                    break
                        if not added_in_this_round:
                            break
                        k += 1
            else:
                selected_terminals = []
                for parent, children in parent_to_children.items():
                    if parent == correct_leaf_parent:
                        selected_terminals.append(correct_leaf)
                    else:
                        selected_terminals.append(random.choice(children))
                        
                if len(selected_terminals) < num_traces:
                    k = 0
                    while len(selected_terminals) < num_traces:
                        added_in_this_round = False
                        for parent, children in parent_to_children.items():
                            if k < len(children) and children[k] not in selected_terminals:
                                selected_terminals.append(children[k])
                                added_in_this_round = True
                                if len(selected_terminals) >= num_traces:
                                    break
                        if not added_in_this_round:
                            break
                        k += 1
        return selected_terminals

