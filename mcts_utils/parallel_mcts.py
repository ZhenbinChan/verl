from __future__ import annotations
import Levenshtein

"""

Implements the MCTS + Self-Refine algorithm from
`Accessing GPT-4 level Mathematical Olympiad Solutions via Monte
Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report`
by Zhang et. al.

The authors' [repo](https://github.com/trotsky1997/MathBlackBox) uses critiques,
refinements, and parent nodes' answers as conversation history.
I haven't tried it yet.

"""

from typing import Optional,Callable, Dict, Any, List
import random
import math
from collections import deque
from enum import Enum
from pydantic import BaseModel,PrivateAttr
from tqdm import tqdm
import json
import time
from functools import partial
from multiprocess import Queue, Process
import yaml
import torch
from typing import List
import re
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
    get_fol_reward,
    check_step_format,
    model_reward_generation,
    query_openai_api_completions_ids_slow,
    query_openai_api_completions_ids_fast,
)

from vllm import LLM, SamplingParams
import os

import numpy as np
import os
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import asyncio

ROOT_UCT_SCORE = 10_000
QUEUE_SIZE = 10000
NUM_PROCESS = 50
from transformers import AutoTokenizer

from collections import Counter
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

def split_and_filter(text, split_chars):
    """
    根据指定的分割符分割字符串，并去除空白部分

    :param text: 要分割的字符串
    :param split_chars: 分割符列表
    :return: 分割后的子字符串列表
    """
    import re
    regex_pattern = '|'.join(map(re.escape, split_chars))
    parts = re.split(regex_pattern, text)
    return [part.strip() for part in parts if part.strip()]


def similarity(str1, str2, split_chars, threshold=0.3, min_proportion=0.65):
    """
    检查两个字符串的部分拼接是否过于相似

    :param str1: 第一个字符串
    :param str2: 第二个字符串
    :param split_chars: 分割符列表，用于分割字符串成片段
    :param threshold: 相似度阈值，表示Levenshtein距离除以较长字符串长度的比例，默认值为0.3
    :param min_proportion: 最小比例，表示拼接部分占原字符串长度的最小比例，默认值为0.5
    :return: 如果两个字符串的部分拼接过于相似，返回True；否则返回False
    """
    def calculate_similarity_ratio(s1, s2):
        """
        计算两个字符串的相似度比率

        :param s1: 第一个字符串
        :param s2: 第二个字符串
        :return: 相似度比率
        """
        len1 = len(s1)
        len2 = len(s2)

        # 计算Levenshtein距离
        lev_distance = Levenshtein.distance(s1, s2)

        # 计算相似比例，取两个字符串长度的较大值
        similarity_ratio = lev_distance / max(len1, len2)

        return similarity_ratio

    len1 = len(str1)
    len2 = len(str2)

    parts1 = split_and_filter(str1, split_chars)
    parts2 = split_and_filter(str2, split_chars)

    # 遍历第一组子串的所有可能组合
    for i in range(len(parts1)):
        for j in range(i+1, len(parts1)+1):
            substring1 = ''.join(parts1[i:j])
            length1_proportion = len(substring1) / len1
            for k in range(len(parts2)):
                for l in range(k+1, len(parts2)+1):
                    substring2 = ''.join(parts2[k:l])
                    length2_proportion = len(substring2) / len2

                    if length2_proportion >= min_proportion or length1_proportion >= min_proportion:
                        if calculate_similarity_ratio(substring1, substring2) < threshold:
                            # print("str1: ", substring1, "\nstr2: ", substring2, "\nratio1:", length1_proportion,
                            #       "\nratio2:", length2_proportion, calculate_similarity_ratio(substring1, substring2))
                            return True

    return False

def similarity_naive(str1,str2):
    # if str1 == str2:
    #     with open("/workspace/lurui/openrlhf-mcts/data/similarity.log", "a") as f:
    #         f.write(f"similarity action\n")
    # else:
    #     with open("/workspace/lurui/openrlhf-mcts/data/similarity.log", "a") as f:
    #         f.write(f"new action\n")
    return str1.strip() == str2.strip()

from graphviz import Digraph

def visualize_tree(root, filename="search_tree"):
    """
    将 TreeNode 结构可视化为 PDF/PNG
    Args:
        root: 你的 TreeNode 根节点
        filename: 输出文件名（不含后缀）
    """
    dot = Digraph(comment='Search Tree', format='png')
    dot.attr(rankdir='TB')  # 从上到下布局 (Top to Bottom)

    # 递归遍历节点
    def add_node_edges(node, parent_id=None):
        # 为当前节点生成唯一ID（使用内存地址或自定义ID）
        node_id = str(id(node))  # 或用 node.id 如果自定义了
        
        # 节点标签：显示 token 或 value
        # 根据你的 TreeNode 属性调整（例如 node.token, node.text, node.value
        answer = getattr(node, 'answer', 'None')
        numSegments = answer.count("\n\n") + 1 if isinstance(answer, str) else 0
        # isCorrect = getattr(node, 'main_chain', False)
        ratio = node.correct_terminal_in_subtree / node.terminal_in_subtree
        label = f"ratioCorrect: {ratio}\nnumSegments: {numSegments}"
        # 可选：显示 Q值或访问次数
        # if hasattr(node, 'visits'):
        #     label += f"\nVisits: {node.visits}"
        # if hasattr(node, 'value'):
        #     label += f"\nValue: {node.value:.2f}"
        # if hasattr(node, 'answer_token'):
        #     label += f"\nTokens: { len(getattr(node, 'answer_token', []))}"
        

        dot.node(node_id, label=label)
        
        # 如果是根节点，没有父节点连接
        if parent_id is not None:
            dot.edge(parent_id, node_id)
        
        # 递归处理子节点
        for child in getattr(node, 'children', []):
            add_node_edges(child, node_id)

    add_node_edges(root)
    dot.render(filename, cleanup=True, view=True)  # view=True 会自动打开图片
    print(f"Tree visualized to {filename}.png")


class MCTSNode(BaseModel):
    state: List[int]
    answer: str
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
    node_id: int = 0
    is_correct: Optional[bool] = None

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, value={self.value:.2f}, visits={self.visits})"

    def __eq__(self, other):
        if isinstance(other, MCTSNode):
            # return self.state == other.state and self.answer == other.answer
            return self.node_id == other.node_id
        return False

    def __hash__(self):
        return hash((self.node_id))

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)

class SelectionPolicy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3

class EvaluationStrategy(Enum):
    SELF_EVALUATION = 1
    MODEL_EVALUATION = 2
    FOL_EVALUATION = 3
    NLI_EVALUATION = 4
    RANDOM_EVALUATION = 5
    TOKEN_ENTROPY = 6
    SENTENCE_ENTROPY = 7

class InitializeStrategy(Enum):
    ZERO_SHOT = 1
    DUMMY_ANSWER = 2


class MCTSr(BaseModel):
    problem: str
    golden_answer: str
    max_nodes: int
    max_token_num: int
    max_time_use: int
    exploration_constant: float = 1.0
    epsilon: float = 1e-10
    reward_limit: int = 95
    excess_reward_penalty: int = 5
    max_depth: int = 40
    max_children: int = 3
    min_children: int = 2
    pass_k: int = 4
    path_num :int = 16
    total_token_num: int = 0
    backbone: str = "glm"
    passed_passktest : bool = False
    backprop: bool = True
    max_node_per_depth  :int = 16
    selection_policy: SelectionPolicy = SelectionPolicy.IMPORTANCE_SAMPLING
    first_token_temperature: bool = False
    selected_terminals: list[MCTSNode] = []
    leaf_num_and_token: dict[int, int] = {}
    # initialize_strategy: InitializeStrategy = InitializeStrategy.ZERO_SHOT

    root: MCTSNode = MCTSNode(
        state=[], answer="I don't know.", max_children=max_children
    )
    temperature: float = 0.9
    top_p: float = 0.9
    llms : List
    tokenizer: Any
    tokenize_fn :Callable 
    detokenize_fn :Callable

    # Logs
    rewards: list[float] = []
    selected_nodes: list[MCTSNode] = []

    depth_node_count: dict[int, int] = {}
    look_ahead :int = 0
    
    # parallel
    concurrent_num: int = 4  # 并行任务数
    node_number :int = 0
    leaf_num :int = 0
    terminal_flag :bool = False
    _lock: Optional[threading.Lock] = PrivateAttr(default=None)
    _childnum_lock: Optional[threading.Lock] = PrivateAttr(default=None)
    prompt_max_len:int = 1024
    leaves: List[int] = []
    step_level_norm: bool = True
    random_pick :bool = False
    use_weighted_value: bool = False
    use_orm_reward: bool = False
    select_correct_leaf: bool = False
    leaf_num_count: int = 1
    use_chain_reward: bool = False
    use_state_value_reward: bool = False
    use_pure_RM:bool = False
    use_pure_binary:bool = False
    shallow_enwide: bool = False
    system_prompt :Optional[str] = None
    average_one_generation:bool = False
    use_value_only:bool = False
    eos_tokens_set :List[int] = [151329,151336,151338]
    a :float = 0.5
    b :float = -2.898

    # custom
    use_api_generation: bool = False
    enable_info: bool = False
    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.SELF_EVALUATION
    check_step_validity: bool = False

    def info(self, message: str, enable_info: Optional[bool] = None):
        enable_info = self.enable_info if enable_info is None else enable_info
        if enable_info:
            print(message)

    def parallel_evaluate_sync(self, children_map, max_workers: int = 8):
        results = []
    
        # 收集所有需要评估的子节点
        all_evaluation_tasks = []  # 存储 (node_index, child_index, child, is_terminated)
        node_children_map = {}  # 存储节点到子节点列表的映射
        
        for node_idx, (node, childrens) in enumerate(children_map.items()):
            node_children_map[node_idx] = (node, childrens)
            
            for child_idx, child in enumerate(childrens):
                is_terminated = child.terminal
                if is_terminated or not self.random_pick:
                    all_evaluation_tasks.append((node_idx, child_idx, child, is_terminated))
        
        if not all_evaluation_tasks:
            return results
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有评估任务
            future_to_task = {}
            for node_idx, child_idx, child, is_terminated in all_evaluation_tasks:
                future = executor.submit(self.self_evaluate, child, is_terminated)
                future_to_task[future] = (node_idx, child_idx, is_terminated)
            
            # 使用字典存储结果，方便后续处理
            all_results = {}
            
            # 收集所有结果
            for future in as_completed(future_to_task):
                node_idx, child_idx, is_terminated = future_to_task[future]
                
                try:
                    reward = future.result()
                    all_results[(node_idx, child_idx)] = (reward, is_terminated)
                except Exception as e:
                    all_results[(node_idx, child_idx)] = (0, is_terminated)
        end_time = time.time()
        self.info(f"Parallel evaluation completed in {end_time - start_time:.2f} seconds")
        
        # 按节点分组处理结果
        node_results = defaultdict(lambda: {"terminal": [], "non_terminal": [], 
                                            "term_indices": [], "non_term_indices": []})
        
        for (node_idx, child_idx), (reward, is_terminated) in all_results.items():
            node_info = node_results[node_idx]
            
            if is_terminated:
                node_info["terminal"].append((child_idx, reward))
                node_info["term_indices"].append(child_idx)
            else:
                node_info["non_terminal"].append((child_idx, reward))
                node_info["non_term_indices"].append(child_idx)
        
        # 处理每个节点
        for node_idx, (node, childrens) in node_children_map.items():
            node_info = node_results[node_idx]
            
            if self.random_pick:
                # 只处理终端节点
                for child_idx, reward in node_info["terminal"]:
                    childrens[child_idx].R = reward
                    childrens[child_idx].value = reward
            else:
                # 处理终端节点
                for child_idx, reward in node_info["terminal"]:
                    childrens[child_idx].R = reward
                    childrens[child_idx].value = reward
                
                # 处理非终端节点（归一化）
                if node_info["non_terminal"]:
                    non_term_rewards = [reward for _, reward in node_info["non_terminal"]]
                    non_term_indices = [idx for idx, _ in node_info["non_terminal"]]
                    
                    if non_term_rewards:
                        non_term_rewards = np.array(non_term_rewards)
                        normalized_rewards = (non_term_rewards - np.mean(non_term_rewards)) / (np.std(non_term_rewards) + 1e-8)
                        
                        for idx, reward in zip(non_term_indices, normalized_rewards):
                            childrens[idx].R = reward
                            childrens[idx].value = reward
            
            assert len(childrens) <= node.max_children, f"Too many children, {len(childrens)} > {node.max_children}"
            node.max_children = len(childrens)
            node.children = childrens
            results.append((node, len(node.children)))
        
        return results

    def self_evaluate(self, node: MCTSNode, is_terminated: bool):
        if is_terminated:
            if node.repeat:
                reward = 0
                node.is_correct = False
            else:
                _, result = check_result(self.problem, node.answer, self.golden_answer)
                node.is_correct = result

                if result:
                    if self.evaluation_strategy == EvaluationStrategy.SELF_EVALUATION:
                        reward = self_reward_generation(self.llms[0], self.problem, node.aggregate_answer, self.tokenize_fn, is_terminated)
                    elif self.evaluation_strategy == EvaluationStrategy.MODEL_EVALUATION:
                        reward = model_reward_generation(self.problem, node.aggregate_answer, is_terminal=is_terminated)
                    elif self.evaluation_strategy == EvaluationStrategy.FOL_EVALUATION:
                        reward = get_fol_reward(self.problem, node.aggregate_answer)
                    else:
                        reward = float(random.randint(0, 1))
                else:
                    reward = 0

            if self.use_pure_binary:
                # self.info("use pure binary")
                node.accumulated_value = reward
            else:
                value = get_qwen_remote_reward_model_value(
                    urls= RM_URLS, question = self.problem, response = node.aggregate_answer)
                if self.use_pure_RM:
                    a = self.a
                    b = self.b
                    x = a*(value-b)
                    result = 1/(1+math.exp(-x))
                    print("rm_score",value, result)
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
                if self.evaluation_strategy == EvaluationStrategy.SELF_EVALUATION:
                    reward = self_reward_generation(self.llms[0], self.problem, node.aggregate_answer, self.tokenize_fn, is_terminated)
                elif self.evaluation_strategy == EvaluationStrategy.MODEL_EVALUATION:
                    reward = model_reward_generation(self.problem, node.aggregate_answer, is_terminal=is_terminated)
                elif self.evaluation_strategy == EvaluationStrategy.FOL_EVALUATION:
                    reward = get_fol_reward(self.problem, node.aggregate_answer)
                else:
                    reward = float(random.randint(0, 1))
            else:
                reward = 0
            
        return reward
                        
    def backpropagate(self, node: MCTSNode,gamma=0.9,main_chain=False):
        self.info("backpropagate")
        parent = node.parent
        node.visits += 1
        if main_chain:
            node.main_chain = True
        # 遍历所有children
        if node.children:
            nume, deno = 0, 0
            for child in node.children:
                reward = child.R - node.R
                q_value = reward + gamma * child.value # 动作价值 = 奖励 + 折扣因子 * 子节点状态价值
                nume += q_value * child.visits
                deno += child.visits
            if deno:
                node.value = nume / deno # 分子 / 分母, 状态价值 = 所有动作价值的加权平均，权重是访问次数
            else:
                self.info("Fail to process value", nume, deno)
        else:
            node.value = node.R
        if parent:
            self.backpropagate(parent,gamma,main_chain)
    
    def leaf_backpropagate(self, node: MCTSNode):
        if node.terminal and node.main_chain:
            node.terminal_in_subtree += 1
            node.correct_terminal_in_subtree += 1
            # 所有父亲的terminal_in_subtree和correct_terminal_in_subtree都加1
            parent = node.parent
            while parent:
                parent.terminal_in_subtree += 1
                parent.correct_terminal_in_subtree += 1
                parent.accumulated_value += node.accumulated_value
                parent = parent.parent
        elif node.terminal:
            node.terminal_in_subtree += 1
            # 所有父亲的terminal_in_subtree都加1
            parent = node.parent
            while parent:
                parent.terminal_in_subtree += 1
                parent.accumulated_value += node.accumulated_value
                parent = parent.parent
    
    def selected_backpropagate(self, node: MCTSNode):
        node.selected_terminal_in_subtree += 1
        # 所有父亲的terminal_in_subtree都加1
        parent = node.parent
        while parent:
            parent.selected_terminal_in_subtree += 1
            parent = parent.parent

    def uct(self, node: MCTSNode, offset = 1):
        if not node.parent:
            return ROOT_UCT_SCORE

        return (node.value+offset+1e-8)/2 + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        ) # NOTE: 不是node.value，而是(node.value+1)/2

    def is_fully_expanded(self, node: MCTSNode):
        next_depth = node.depth + 1
        next_depth_count = self.depth_node_count.get(next_depth, 0)
        if next_depth_count >= self.max_node_per_depth: # 是否超过每一层的节点数限制
            return True
        return len(node.children) >= node.max_children or node.terminal # 是否超过每个节点的孩子数限制，或者是否是终止节点

    def weighted_sample_no_replacement(self, candidates, weights, k):
        total_candidates = len(candidates)
        k = min(total_candidates, k)  # 确保 k 不超过候选项数量

        if k >= total_candidates:
            return list(range(total_candidates))  # 如果 k 大于等于候选项数量，返回所有索引
        
        if k <= 0 or total_candidates == 0:
            return []  # 如果 k <= 0 或候选项为空，返回空列表

        indices = list(range(total_candidates))
        selected_indices = []

        for _ in range(k):
            # 根据权重随机选择一个索引
            try:
                chosen_index = random.choices(indices, weights=weights, k=1)[0]
            except ValueError:
                # 如果在选择索引时发生错误，重新计算权重再试一次
                uct_scores = [self.uct(node, 2) for node in candidates]
                chosen_index = random.choices(indices, weights=uct_scores, k=1)[0]
                
            selected_indices.append(chosen_index)
            
            # 移除已选择的索引和对应的权重
            remove_index = indices.index(chosen_index)
            indices.pop(remove_index)
            weights.pop(remove_index)
            candidates.pop(remove_index)
                
        return selected_indices

    def select_node(self, k=1,random_pick=False):
        """Select up to k non-fully expanded nodes with the highest UCT value.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own
        """
        candidates = []
        to_consider = deque([self.root])

        while to_consider:
            current_node = to_consider.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)

        if len(candidates) == 0:
            for node in to_consider:
                assert self.is_fully_expanded(node), "Not fully expanded"

        if not candidates:
            return None

        selected_nodes = []
        if random_pick:
            selected_nodes = random.sample(candidates, min(k, len(candidates)))
            return selected_nodes
        else:
            if self.selection_policy == SelectionPolicy.GREEDY:
                selected_nodes = sorted(candidates, key=self.uct, reverse=True)[:k]

            elif self.selection_policy == SelectionPolicy.IMPORTANCE_SAMPLING:
                uct_scores = [self.uct(node) for node in candidates]
                # 拷贝一份candidates，避免修改原列表
                candis = candidates.copy()
                selected_indices = self.weighted_sample_no_replacement(candis, uct_scores, k)
                selected_nodes = [candidates[i] for i in selected_indices]

            elif self.selection_policy == SelectionPolicy.PAIRWISE_IMPORTANCE_SAMPLING:
                uct_scores = [self.uct(node) for node in candidates]
                pairs = [
                    (i, j) for i in range(len(candidates)) for j in range(len(candidates))
                ]
                pair_weights = [
                    max(uct_scores[i], uct_scores[j]) -
                    min(uct_scores[i], uct_scores[j])
                    for i, j in pairs
                ]
                selected_pair_indices = random.choices(
                    range(len(pairs)), weights=pair_weights, k=min(k, len(pair_weights))
                )

                for pair_idx in selected_pair_indices:
                    selected_candidate_idx = max(
                        pairs[pair_idx], key=lambda x: uct_scores[x]
                    )
                    if candidates[selected_candidate_idx] not in selected_nodes:
                        selected_nodes.append(candidates[selected_candidate_idx])
                    if len(selected_nodes) >= k:
                        break
            else:
                raise ValueError(
                    f"Invalid selection policy: {self.selection_policy}")

            return selected_nodes

    def initialize(self, add_understanding=False):
        """Generate a zero-shot answer."""
        if not self.use_api_generation:
            init_state = self.tokenize_fn(
                [[self.problem],[None]], self.prompt_max_len, device="cpu",system_prompt=self.system_prompt
            )["input_ids"][0]
        else:
            init_state = []

        self.root = MCTSNode(
            state=init_state,
            answer="",
            max_children=self.max_children,
            depth=0
        )
        self.depth_node_count[0] = 1  # Initialize for root node
    
    # @profile_with_time(output_dir="run_function_profile_yujiang")
    def run(self):
        self.initialize()
        node_number = 0
        leaf_num = 0
        start_time = time.time()
        leaf_rank = 0

        while node_number < self.max_nodes and time.time() - start_time < self.max_time_use:
            nodes = self.select_node(k=self.concurrent_num, random_pick=self.random_pick) # 根据每一个MCTSNode的UCT值选择要扩展的节点，返回一个列表
            if not nodes:
                self.info("terminated because no node to expand")
                break
            results = self.expand(nodes)

            # ------------------------------ 用于统计 ------------------------------
            self.total_token_num += results[1]
            terminal_generated = 0
            for node, child_num in results[0]:
                for child in node.children:
                    is_terminated = child.terminal
                    if is_terminated:
                        terminal_generated += 1
            leaf_rank += terminal_generated
            self.leaf_num_and_token[leaf_rank] = self.total_token_num
            # ----------------------------------------------------------------------

            for node, child_num in results[0]:
                if child_num == 0:
                    self.info(f"Cannot expand the node {node}")
                    continue
                else:
                    node_number += child_num 
                    # 用于统计每一层的节点数, 在select_node函数中会用到这个统计来判断是否超过每一层的节点数限制
                    self.depth_node_count[node.depth + 1] = self.depth_node_count.get(node.depth + 1, 0) + child_num
                    
                    child_terminal_exits = False
                    for child in node.children:
                        is_terminated = child.terminal
                        if is_terminated and child.is_correct:
                            # assert child.R == 1, "correct leaf reward is not 1"
                            child.main_chain = True
                            if self.backprop: # 是否进行回传，如果进行回传，正确的叶子节点会将价值回传给父亲节点，父亲节点的价值会更新，进而影响UCT值
                                self.backpropagate(child, child.main_chain)
                        else:
                            if self.backprop:
                                self.backpropagate(child)
                        if is_terminated:
                            self.leaves.append(child)

                        if is_terminated and not child_terminal_exits:
                            leaf_num += 1
                            child_terminal_exits = True

            if leaf_num >= self.pass_k:
                self.info(f"terminated because reach {leaf_num} leaf nodes")
                break
        
        self.leaf_normalize(self.leaves) # leaf.accumulated_value = leaf.accumulated_value - mean[i] (leave-one-out normalization)
        for leaf in self.leaves:
            self.leaf_backpropagate(leaf) # parent.accumulated_value += node.accumulated_value
        if self.average_one_generation:
            self.update_accumulated_values() # node.accumulated_value = total_value / terminal_children
        self.select_terminal()
        if self.use_weighted_value:
            for leaf in self.selected_terminals:
                self.selected_backpropagate(leaf)
            self.weighted_update()
    
    def multi_language(self,text):
        """检查是否包含非英语内容（中文、日文、韩文、俄语）"""
        # 定义字符集范围
        chinese_range = re.compile(r'[\u4e00-\u9fff]')  # 中文
        japanese_range = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')  # 日文假名
        korean_range = re.compile(r'[\uac00-\ud7af]')  # 韩文
        russian_range = re.compile(r'[а-яА-ЯёЁ]')  # 俄语（包括大小写）
        # 检查是否包含非英语字符
        has_chinese = chinese_range.search(text)
        has_japanese = japanese_range.search(text)
        has_korean = korean_range.search(text)
        has_russian = russian_range.search(text)
        return has_chinese or has_japanese or has_korean or has_russian
        
    # @profile_with_time(output_dir="run_function_profile_yujiang")
    def expand(self, nodes):
        if len(nodes) == 0:
            return [], 0
        stops = get_stops(self.backbone) # '<|im_end|>', '\n\n', ']\n\n'. '.\n\n', ':\n\n', '*\n\n', ')\n\n', '>\n\n'
        all_children_token_num = 0
        max_tokens_per_step = self.max_token_num
        max_attempts = 3
        children_map = {node: [] for node in nodes}  # 用于记录每个节点的孩子

        attempts = 0
        while attempts < max_attempts:
            prompts, node_prompts_map = [], []

            for node in nodes:
                num_current_children = len(children_map[node])
                prompts.extend([node.state] * (node.max_children - num_current_children))
                node_prompts_map.extend([node] * (node.max_children - num_current_children))
            if not prompts:
                break  # 如果没有需要生成的孩子，退出

            attempts += 1
            
            next_tokens = []
            next_strs = []
            # 第一 Token 采样（可选的 first_token_temperature） 旨在增加搜索树的宽度多样性。
            # 逻辑：如果开启且随机命中，它会先单独采样第一个 Token，并使用较高的 Temperature。
            # 目的：在大模型推理中，第一个 Token 往往决定了整句话的逻辑走向。通过这种方式，强制模型从不同的“念头”开始推理，防止搜索树过于坍缩
            if self.first_token_temperature and random.random() < 0.5 and not self.use_api_generation:
                self.info("using first token tempeature")
                first_tokens = top_k_sampling(
                    llm = self.llms[0],
                    prompts = prompts,
                    top_p = self.top_p,
                    skip_special_tokens=False,
                    stops=stops,
                )
                next_tokens = [random.choice(used_logprobs) for used_logprobs in first_tokens]
                next_strs = [self.detokenize_fn([next_token]) for next_token in next_tokens]
                prompts = [prompt + [next_token] for prompt, next_token in zip(prompts, next_tokens)]
            start_time = time.time()
            if not self.use_api_generation:
                responses_token, responses_str, finish_reasons, stop_tokens, token_nums = query_local_vllm_completions_ids(
                    prompts,
                    llm=self.llms[0],
                    n=1,
                    skip_special_tokens=True,
                    max_tokens=max_tokens_per_step,
                    stops=stops,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    model=self.backbone,
                )
            else:
                messages=[[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.problem},
                    {"role": "assistant", "content": self.detokenize_fn(prompt)}
                ] for prompt in prompts]
                responses_token, responses_str, finish_reasons, stop_tokens, token_nums = query_openai_api_completions_ids_fast(
                    messages,
                    n=1,
                    max_tokens=max_tokens_per_step,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stops=['\n\n'],
                    tokenizer=self.tokenizer
                )
            end_time = time.time()
            self.info(f"Generation latency: {end_time - start_time:.2f} seconds")

            # 生成响应后，遍历每个结果，将其封装成 MCTSNode 对象
            for idx, (response_token_list, response_str_list, finish_reason_list, stop_token_list, token_num_list) in enumerate(zip(responses_token, responses_str, finish_reasons, stop_tokens, token_nums)):
                node = node_prompts_map[idx]
                response_token = response_token_list[0]
                response = response_str_list[0]
                finish_reason = finish_reason_list[0]
                stop_token = stop_token_list[0]
                if next_tokens:
                    assert len(next_strs) != 0 and len(next_tokens) != 0 and len(next_strs) == len(next_tokens), "next tokens and next strs should have the same length"
                    response = next_strs[idx] + response
                    
                    response_token = [next_tokens[idx]] + response_token
                if response is None:
                    self.info("response is None")
                    continue  # 如果响应为空或未结束，跳过

                action = response
                # 在调用 vLLM 时，如果模型命中了你设置的 stops（比如遇到换行符 \n\n 或某个特定的步骤标识符），vLLM 默认不会把这个停止符包含在返回的字符串中
                if not ((stop_token is None) or (stop_token in self.eos_tokens_set)):
                    stop_token_str = self.detokenize_fn([stop_token])
                    action += stop_token_str
                    response_token += [stop_token]

                new_action = action
                new_action_token = response_token

                # 过滤与当前 children_map[node] 中的项重复
                # existing_actions = [
                #     child.answer
                #     for child in children_map[node]
                # ]

                # if any(
                #     similarity(new_action, existing_action, split_chars=["\n\n", ". "])
                #     for existing_action in existing_actions
                # ):
                #     continue  # 如果新的动作与现有的孩子过于相似，跳过

                # if any(
                #     similarity_naive(new_action, existing_action)
                #     for existing_action in existing_actions
                # ):
                #     continue  # 如果新的动作与现有的孩子过于相似，跳过

                expanded_state = node.state + response_token
                new_aggregate_answer = node.aggregate_answer + new_action
                new_aggregate_answer_token = node.aggregate_answer_token + new_action_token
                all_children_token_num += token_num_list[0]  # 假设我们使用列表中的第一个标记数

                if (len(new_aggregate_answer_token) > self.max_token_num) or (find_repeated_patterns(new_aggregate_answer)):
                    repeat = True
                else:
                    repeat = False

                if (stop_token is None) or (stop_token in self.eos_tokens_set) or repeat:
                    if_finish = True
                else:
                    if_finish = False

                finished = self.judge_finished(
                    if_finish, node.depth + 1
                )
                if len(new_aggregate_answer_token) > self.max_token_num:
                    self.info("terminal because exceed max token num")
                elif find_repeated_patterns(new_aggregate_answer):
                    self.info("terminal because find repeated patterns")
                elif finished:
                    self.info("terminal because finished")

                if self.shallow_enwide:
                    self.info("shallow enwide")
                    max_children = max(node.max_children / 2, self.min_children)
                else:
                    max_children = node.max_children

                child_node = MCTSNode(
                    state=expanded_state,
                    answer=new_action,
                    answer_token=new_action_token,
                    aggregate_answer=new_aggregate_answer,
                    aggregate_answer_token=new_aggregate_answer_token,
                    parent=node,
                    depth=node.depth + 1,
                    terminal=finished,
                    max_children=max_children,
                    repeat=repeat,
                    node_id=self.leaf_num_count
                )
                if finished:
                    self.info(f"Generated child node with state: {self.detokenize_fn(child_node.state)}")
                self.leaf_num_count += 1

                children_map[node].append(child_node)

            if all(len(children_map[node]) >= node.max_children for node in nodes):
                break  # 如果所有节点的孩子都生成完毕，退出

        self.info(f"all childrens: {sum(len(children) for children in children_map.values())}")

        # for node, childrens in children_map.items():
        #     if self.random_pick:
        #         for i, child in enumerate(childrens):
        #             if child.terminal:
        #                 child.R = self.self_evaluate(child, True)
        #                 child.value = child.R
        #     else:
        #         non_leaf_rewards = []
        #         non_leaf_indexes = []
        #         for i, child in enumerate(childrens):
        #             if child.terminal:
        #                 child.R = self.self_evaluate(child, True)
        #                 child.value = child.R
        #             else:
        #                 reward = self.self_evaluate(child, False)
        #                 non_leaf_rewards.append(reward)
        #                 non_leaf_indexes.append(i)

        #         # 归一化到（0，1）之间， 消除 PRM 打分的绝对偏置（Bias），只保留相对的好坏
        #         if non_leaf_rewards:
        #             non_leaf_rewards = np.array(non_leaf_rewards)
        #             normalized_rewards = (non_leaf_rewards - np.mean(non_leaf_rewards)) / (np.std(non_leaf_rewards) + 1e-8)

        #             for i, reward in zip(non_leaf_indexes, normalized_rewards):
        #                 childrens[i].R = reward
        #                 childrens[i].value = reward

        #     assert len(childrens) <= node.max_children, f"Too many children, {len(childrens)} > {node.max_children}"
        #     node.max_children = len(childrens) # 已经扩展过的节点不再扩展更多的孩子
        #     node.children = childrens

        #     results.append((node, len(node.children)))

        results = self.parallel_evaluate_sync(children_map)

        return results, all_children_token_num

    def print(self):
        print_tree(self.root)

    def judge_finished(self, is_stopped, depth):
        finished = is_stopped or depth > self.max_depth
        return finished

    def normalize_backprop(self):
        # 对所有叶子向上更新
        for node in self.selected_terminals:
            parent = node.parent
            while parent:
                parent.accumulated_value += node.accumulated_value
                parent.visited_terminal += 1
                if parent.visited_terminal == parent.terminal_in_subtree:
                    parent.accumulated_value = parent.accumulated_value / parent.terminal_in_subtree
                parent = parent.parent
        self.normalize_all_steps()

    def normalize_all_steps(self):
        # 从root开始遍历所有节点，对所有terminal_in_subtree！=0节点的accumulated_value进行归一化
        all_steps = []
        to_consider = deque([self.root])
        while to_consider:
            current_node = to_consider.popleft()
            if current_node.terminal_in_subtree != 0 or current_node.terminal:
                all_steps.append(current_node)
            to_consider.extend(current_node.children)

        self.info(f"all_step value: {[node.accumulated_value for node in all_steps]}, {len(all_steps)}")
        if self.step_level_norm:
            step_sum = 0
            step_num = 0
            for node in all_steps:
                step_sum += node.accumulated_value*node.terminal_in_subtree
                step_num += node.terminal_in_subtree
            if step_num == 0:
                mean = 0
            else:
                mean = step_sum/step_num
            self.info("mean:", mean,step_sum,step_num)
            for node in all_steps:
                node.accumulated_value = node.accumulated_value - mean
        else:
            # self.info("token level normalization")
            step_sum = 0
            step_num = 0
            for node in all_steps:
                step_sum += node.accumulated_value*node.terminal_in_subtree*len(node.answer_token)
                step_num += node.terminal_in_subtree*len(node.answer_token)
            if step_num == 0:
                mean = 0
            else:
                mean = step_sum/step_num
            # self.info("mean:", mean)
            for node in all_steps:
                node.accumulated_value = node.accumulated_value - mean
                

    def leaf_normalize(self,nodes):
        leaf_correctness = [leaf.accumulated_value for leaf in nodes]
        # self.info("leaf_correctness",leaf_correctness)
        _sum = sum(leaf_correctness)
        num = len(leaf_correctness) - 1
        if num == 0:
            return
        else:
            mean = [(_sum - leaf_correctness[i]) / num for i in range(len(leaf_correctness))]
            for i, leaf in enumerate(nodes):
                leaf.accumulated_value = leaf.accumulated_value - mean[i] # 类似于 leave-one-out 的方式，计算每个叶子节点的平均正确率（不包括它自己），然后用这个平均正确率来归一化这个叶子节点的 accumulated_value。这种方法可以减少某个叶子节点对自己的评价的影响，使得评价更加客观。
        # self.normalize_backprop()

    def compute_accumulated_value(self,node: MCTSNode):
        if not node.children:  # If the node is a leaf node
            return node.accumulated_value

        # Post-order traversal: first process all children
        total_value = 0
        terminal_children = 0
        for child in node.children:
            if child.terminal_in_subtree > 0:
                terminal_children += 1
                total_value += self.compute_accumulated_value(child)
        self.info("children value",total_value,terminal_children)

        # Calculate the average accumulated value for the current node
        node.accumulated_value = total_value / terminal_children if terminal_children > 0 else 0
        return node.accumulated_value

    
    # Helper function to initialize calculation from the root node
    def update_accumulated_values(self):
        self.compute_accumulated_value(self.root)
        
    def compute_weighted_update(self, node: MCTSNode):
        node.accumulated_value = node.accumulated_value * node.terminal_in_subtree / node.selected_terminal_in_subtree
        for child in node.children:
            self.compute_weighted_update(child)
        
    def weighted_update(self):
        self.compute_weighted_update(self.root)


    def select_terminal(self):
        # 从self.leaves中选择self.path_num 个叶子节点, 尽可能挑选同样数量的正确和错误的叶子，同一个父亲的叶子如果同对同错，只能选一个
        parent_to_children = {}

        if len(self.leaves) < 3:
            return False
        
        correct_leaf_parent = None
        correct_leaf = None
        for leaf in self.leaves:
            if leaf.main_chain:
                correct_leaf_parent = leaf.parent
                correct_leaf = leaf
            parent = leaf.parent
            if parent not in parent_to_children.keys():
                parent_to_children[parent] = []
            parent_to_children[parent].append(leaf)
        
        total_sum = len(parent_to_children.keys())
        if correct_leaf_parent is not None:
            assert correct_leaf is not None, "correct leaf is None"
            self.info("got correct leaf!")
        
        if not self.select_correct_leaf:
            self.info("do not manually select correct leaf")
            correct_leaf = None
            correct_leaf_parent = None

        if total_sum == self.path_num:
            if correct_leaf is None:
                selected_terminals = []
                # 为每个父节点选择一个孩子
                for parent, children in parent_to_children.items():
                    selected_terminals.append(random.choice(children))
                self.selected_terminals = selected_terminals
                return True
            else:
                selected_terminals = []
                # 为每个父节点选择一个孩子
                for parent, children in parent_to_children.items():
                    if parent == correct_leaf_parent:
                        selected_terminals.append(correct_leaf)
                    else:
                        selected_terminals.append(random.choice(children))
                self.selected_terminals = selected_terminals
                return True

        elif total_sum > self.path_num:
            if correct_leaf is None:
                # 首先随机选self.path_num个父节点
                selected_parents = random.sample(list(parent_to_children.keys()), self.path_num) # 不能对set进行采样
                selected_terminals = []
                for parent in selected_parents:
                    selected_terminals.append(random.choice(parent_to_children[parent]))
                self.selected_terminals = selected_terminals
                return True
            else:
                other_parents = [parent for parent in parent_to_children.keys() if parent != correct_leaf_parent]
                selected_parents = random.sample(other_parents, self.path_num - 1)
                selected_terminals = [correct_leaf]
                for parent in selected_parents:
                    selected_terminals.append(random.choice(parent_to_children[parent]))
                self.selected_terminals = selected_terminals
                return True     
        else:
            if correct_leaf is None:
                selected_terminals = []
                # 为每个父节点选择一个正确的孩子和一个错误的孩子（如果有的话）
                for parent, children in parent_to_children.items():
                    selected_terminals.append(random.choice(children))
                if len(selected_terminals) < self.path_num:
                    k = 0
                    while len(selected_terminals) < self.path_num:
                        added_in_this_round = False
                        for parent, children in parent_to_children.items():
                            if k < len(children) and children[k] not in selected_terminals:
                                selected_terminals.append(children[k])
                                added_in_this_round = True
                                if len(selected_terminals) >= self.path_num:
                                    break     
                        if not added_in_this_round:
                            break  # 如果这一轮没有添加新路径，则不能继续补全
                        k += 1
                # while len(selected_terminals) < self.path_num:
                #     assert len(selected_terminals) > 0, "Not enough terminal nodes"
                #     # 把selected_terminals shuffle一下，然后再从头开始添加
                #     random.shuffle(selected_terminals)
                #     for node in selected_terminals:
                #         selected_terminals.append(node)
                #         if len(selected_terminals) >= self.path_num:
                #             break
                self.selected_terminals = selected_terminals
                return True
            else:
                selected_terminals = []
                for parent, children in parent_to_children.items():
                    if parent == correct_leaf_parent:
                        selected_terminals.append(correct_leaf)
                    else:
                        selected_terminals.append(random.choice(children))
                if len(selected_terminals) < self.path_num:
                    k = 0
                    while len(selected_terminals) < self.path_num:
                        added_in_this_round = False
                        for parent, children in parent_to_children.items():
                            if k < len(children) and children[k] not in selected_terminals:
                                selected_terminals.append(children[k])
                                added_in_this_round = True
                                if len(selected_terminals) >= self.path_num:
                                    break     
                        if not added_in_this_round:
                            break  # 如果这一轮没有添加新路径，则不能继续补全
                        k += 1
                # while len(selected_terminals) < self.path_num:
                #     assert len(selected_terminals) > 0, "Not enough terminal nodes"
                #     # 把selected_terminals shuffle一下，然后再从头开始添加
                #     random.shuffle(selected_terminals)
                #     for node in selected_terminals:
                #         selected_terminals.append(node)
                #         if len(selected_terminals) >= self.path_num:
                #             break
                self.selected_terminals = selected_terminals
                return True

                
###########################
# Functions for saving and loading the tree
###########################

def print_tree(node: MCTSNode | None, level: int = 0, index: int = 0):
    if node is None:
        return
    indent = " " * level * 2 + str(level) + f"-{index}: "
    node_str = repr(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for index, child in enumerate(node.children):
        print_tree(child, level + 1, index)


def convert_to_json(node: MCTSNode):
    if not node.children:
        return {
            "answer": node.answer,
            "aggregate_answer": node.aggregate_answer,
            "value": node.value,
            "R" : node.R,
            "visits": node.visits,
            "reward_samples": node.reward_samples,
            "depth": node.depth,
            "main_chain": node.main_chain,
            "terminal": node.terminal,
            # "selected_terminal": node.selected_terminal,
            "terminal_in_subtree": node.terminal_in_subtree,
            "correct_terminal_in_subtree": node.correct_terminal_in_subtree,
            "accumulated_value": node.accumulated_value,
            "visited_terminal": node.visited_terminal,
            "node_id": node.node_id,
            "max_children": node.max_children
        }
    else:
        return {
            "answer": node.answer,
            "aggregate_answer": node.aggregate_answer,
            "value": node.value,
            "R" : node.R,
            "visits": node.visits,
            "reward_samples": node.reward_samples,
            "depth": node.depth,
            "main_chain": node.main_chain,
            "terminal": node.terminal,
            # "selected_terminal": node.selected_terminal,
            "children": [convert_to_json(child) for child in node.children],
            "terminal_in_subtree": node.terminal_in_subtree,
            "correct_terminal_in_subtree": node.correct_terminal_in_subtree,
            "accumulated_value": node.accumulated_value,
            "visited_terminal": node.visited_terminal,
            "node_id": node.node_id,
            "max_children": node.max_children
        }


def build_tree_based_on_json(json_data):
    if "children" in json_data:
        node = MCTSNode(**json_data)
        node.children = [build_tree_based_on_json(
            child) for child in json_data["children"]]
        return node
    else:
        return MCTSNode(**json_data)

###########################
# multiprocess workers
###########################

def chain_worker(
    item,
    llm,
    init_prompt,
    prompt_key="problem",
    answer_key="golden_answer",
    args=None,
    first_token_temperature=0,
    detokenize_fn=None,
):
    pass_k = args["path_num"]
    stops = [151336, 151329,151338]
    max_attempts = 3
    attempts = 0
    paths = []
    while attempts < max_attempts:
        prompts = [init_prompt] * (pass_k - len(paths))
        if not prompts:
            break  # 如果没有需要生成的孩子，退出

        attempts += 1
        next_tokens = []
        next_strs = []
        if first_token_temperature:
            print("using first token tempeature")
            first_tokens = top_k_sampling(
                llm = llm,
                prompts = prompts,
                top_p = self.top_p,
                skip_special_tokens=False,
                stops=stops,
            )
            full_strs = [detokenize_fn(first_token) for first_token in first_tokens]
            next_tokens = [random.choice(used_logprobs) for used_logprobs in first_tokens]
            next_strs = [detokenize_fn([next_token]) for next_token in next_tokens]
            print("full_strs", full_strs)
            # with open("/workspace/lurui/openrlhf-mcts/data/first_tokens.jsonl", "a") as f:
            #     f.write(json.dumps({"prompts": prompts, "next_strs": full_strs}))
            #     f.write("\n")
            prompts = [prompt + [next_token] for prompt, next_token in zip(prompts, next_tokens)]

        responses_token, responses_str, finish_reasons, stop_tokens, token_nums = query_local_vllm_completions_ids(
            prompts,
            llm=llm,
            n=1,
            skip_special_tokens=True,
            max_tokens=4096,
            stops=stops,
            temperature=args["temperature"],
            top_p=args["top_p"],
            model="glm",
            min_tokens = 0
        )
        # print("stop_tokens", stop_tokens)

        for idx, (response_token_list,response_str_list, finish_reason_list, stop_token_list, token_num_list) in enumerate(zip(responses_token, responses_str, finish_reasons, stop_tokens, token_nums)):
            response_token = response_token_list[0]
            response = response_str_list[0]
            finish_reason = finish_reason_list[0]
            stop_token = stop_token_list[0]
            
            if next_tokens:
                assert len(next_strs) != 0 and len(next_tokens) != 0 and len(next_strs) == len(next_tokens), "next tokens and next strs should have the same length"
                response = next_strs[idx] + response
                response_token = [next_tokens[idx]] + response_token

            if (attempts != max_attempts) and (response is None):
                continue  # 如果响应为空或未结束，跳过

            action = response if response is not None else ""
            token_action = response_token if response_token is not None else []

            paths.append({"answer": action, "token_answer": token_action})
            if len(paths) >= pass_k:
                break

    assert len(paths) == pass_k, f"Failed to generate {pass_k} paths"

    results = []
    for path in paths:
        path["reward"] = get_qwen_remote_reward_model_value(urls= RM_URLS, question = item[prompt_key], response = path["answer"])
        path["pass_ratio"] = check_result(item[args["prompt_key"]], path["answer"], item[answer_key],checker_urls=EVALUATOR_URLS,extractor_urls=EXTRACTOR_URLS)[-1]
        results.append(path["pass_ratio"])
    _sum = sum(results)
    num = len(results) - 1
    if num == 0:
        return paths
    else:
        mean = [(_sum - results[i]) / num for i in range(len(results))]
        for i, path in enumerate(paths):
            path["value"] = path["pass_ratio"] - mean[i]
        paths = [[path] for path in paths]
        return paths

def mcts_worker(
    item,
    llm, 
    tokenize_fn,
    detokenize_fn,
    prompt_key="problem",
    answer_key="golden_answer",
    args=None,
    system_prompt=None,
):
    # 随机 sleep 一段时间，一分钟以内
    # time.sleep(random.randint(0, 60))
    pid = os.getpid()
    problem = item[prompt_key]
    answer = item[answer_key]
    mcts = MCTSr(
        temperature=args["temperature"],
        top_p=args["top_p"],
        problem=problem,
        golden_answer=answer,
        max_nodes=args["max_nodes"],
        exploration_constant=args["exploration_constant"],
        selection_policy=SelectionPolicy.IMPORTANCE_SAMPLING,
        backbone=args["backbone"],
        max_children=args["max_children"],
        min_children=args["min_children"],
        pass_k=args["pass_k"],
        max_depth=args["max_depth"],
        backprop=args["backprop"],
        max_node_per_depth = args["max_node_per_depth"],
        first_token_temperature=args["first_token_temperature"],
        look_ahead=args["look_ahead"],
        llms=[llm],
        tokenize_fn = tokenize_fn,
        detokenize_fn = detokenize_fn,
        concurrent_num=args["concurrent_num"],
        path_num = args["path_num"],
        prompt_max_len = args["prompt_max_len"],
        max_token_num = args["max_token_num"],
        max_time_use = args["max_time_use"],
        step_level_norm = args["step_level_norm"],
        random_pick = args["random_pick"],
        use_weighted_value = args["use_weighted_value"],
        use_orm_reward = args["use_orm_reward"],
        select_correct_leaf = args["select_correct_leaf"],
        use_chain_reward = args["use_chain_reward"],
        use_state_value_reward = args["use_state_value_reward"],
        use_value_only = args["use_value_only"],
        use_pure_RM = args["use_pure_RM"],
        use_pure_binary = args["use_pure_binary"],
        shallow_enwide = args["shallow_enwide"],
        system_prompt=system_prompt,
        average_one_generation = args["average_one_generation"],
        a = args["a"],
        b = args["b"],
    )
    # print(mcts.max_children)
    start_time = time.time()
    mcts.run()
    try:
        # mcts.run()
        root = mcts.root
        # with open("/workspace/lurui/openrlhf-glm/logs/outputs/trees_vine.jsonl", "a",encoding="utf-8") as f:
        # with open("/workspace/lurui/openrlhf-mcts/data/paths.jsonl", "a",encoding="utf-8") as f:
        #     tree_json = convert_to_json(root)
        #     tree_json["random_pick"] = args["random_pick"]
        #     # tree_json["time_used"] = time_used
        #     json.dump(tree_json, f)
        #     f.write("\n")
        # print("selected_terminals",mcts.selected_terminals[0])
        paths = gather_paths(mcts.root,mcts.selected_terminals,args["path_num"],use_orm_reward = mcts.use_orm_reward,use_chain_reward = mcts.use_chain_reward,step_level_norm = mcts.step_level_norm,use_state_value_reward = mcts.use_state_value_reward,use_value_only = mcts.use_value_only,average_one_generation = mcts.average_one_generation,advantage_mix_allancestor=args["advantage_mix_allancestor"])
        time_used = time.time() - start_time
        pass_num = pass_rate(paths)
        # os.makedirs("logs/outputs", exist_ok=True)
        with open("/workspace/lurui/openrlhf-mcts/data/outputs/trees_mcts.jsonl", "a",encoding="utf-8") as f:
        # with open("/workspace/lurui/openrlhf-mcts/data/paths.jsonl", "a",encoding="utf-8") as f:
            tree_json = convert_to_json(root)
            tree_json["random_pick"] = args["random_pick"]
            tree_json["time_used"] = time_used
            tree_json["args"] = args
            tree_json["total_nodes"] = mcts.leaf_num_count
            tree_json["total_token_num"] = mcts.total_token_num
            tree_json["pass_num"] = pass_num
            tree_json["leaf_num_and_token"] = mcts.leaf_num_and_token
            json.dump(tree_json, f)
            f.write("\n")
    except Exception as e:
        # print(f"Error in MCTS: {e}")
        os.makedirs("logs/outputs", exist_ok=True)
        with open("logs/outputs/error.log", "a") as f:
            f.write(f"Error in MCTS: {e}")
        paths = None
        time_used = time.time() - start_time
    if paths is None:
        os.makedirs("logs/outputs", exist_ok=True)
        with open("logs/outputs/response_type.jsonl", "a",encoding="utf-8") as f:
            f.write("use chain_worker\n")
        # init_prompt = tokenize_fn([[problem],[None]],args["prompt_max_len"], device="cpu",system_prompt=system_prompt)
        init_prompt = tokenize_fn([[problem],[None]],args["prompt_max_len"], device="cpu",system_prompt=system_prompt)["input_ids"][0].tolist()
        paths = chain_worker(item, llm, init_prompt, prompt_key, answer_key, args)
        return paths, init_prompt
    else:
        with open("logs/outputs/response_type.jsonl", "a",encoding="utf-8") as f:
            f.write("use mcts_worker\n")
        return paths,root.state

def get_stops(backbone="glm"):
    if backbone == "glm":
        return [271, 151336, 151329,151338, 2533, 382, 1447, 21467, 692]
    elif backbone == "qwen":
        return [151645, 271, 2533, 382, 1447, 21518, 692, 1339]

def normalize_selected_terminals(selected_terminals: list[MCTSNode]):
    leaf_orm_value = [leaf.accumulated_value for leaf in selected_terminals]
    _sum = sum(leaf_orm_value)
    num = len(leaf_orm_value) - 1
    if num == 0:
        return leaf_orm_value
    else:
        mean = [(_sum - leaf_orm_value[i]) / num for i in range(len(leaf_orm_value))]
        orm_normalized = [leaf_orm_value[i] - mean[i] for i in range(len(leaf_orm_value))]
        return orm_normalized

def fill_in_paths(paths):
    # 对于每个路径，如果存在"value"=0，就用他的前一个节点的"value"填充
    for path in paths:
        for i in range(1,len(path)):
            epsilon = 1e-8
            if abs(path[i]["value"]) < epsilon: 
            # if path[i]["value"] == 0:
                assert i > 0, "value=0 in the first node"
                assert path[i]["value"] < epsilon  and path[i]["value"] > -epsilon, "value is not 0"
                # print("fill in value",path[i-1]["value"])
                path[i]["value"] = path[i-1]["value"]
    return paths

def normalize_all_paths(paths,step_level_norm = False):
    # 对所有路径进行归一化
    if step_level_norm:
        state_value_sum = 0
        state_value_num = 0
        for path in paths:
            for node in path:
                state_value_sum += node["state_value"]
                state_value_num += 1
        if state_value_num == 0:
            mean = 0
        else:
            mean = state_value_sum/state_value_num
        for path in paths:
            for node in path:
                node["state_value"] = node["state_value"] - mean
        return paths
    else:
        state_value_sum = 0
        state_value_num = 0
        for path in paths:
            for node in path:
                state_value_sum += node["state_value"]*len(node["token_answer"])
                state_value_num += len(node["token_answer"])
        if state_value_num == 0:
            mean = 0
        else:
            mean = state_value_sum/state_value_num
        for path in paths:
            for node in path:
                node["state_value"] = node["state_value"] - mean
        return paths

def path_from_root_to_node(node: MCTSNode,average_one_generation:bool = False) -> List[Dict[str, Any]]:
    path = []
    while node.parent is not None:
        if average_one_generation:
            print("average_one_generation when gather")
            parent_value = node.parent.accumulated_value
            child_value = node.accumulated_value
        else:
            parent_value = node.parent.accumulated_value / node.parent.terminal_in_subtree
            child_value = node.accumulated_value / node.terminal_in_subtree
        if node.terminal:
            assert node.terminal_in_subtree == 1, f"terminal_in_subtree is not 1,{node.terminal_in_subtree}"
        path.append(
            {'answer': node.answer,
             'token_answer':node.answer_token,
             'reward': node.value,
             "pass_ratio":node.correct_terminal_in_subtree / node.terminal_in_subtree,
             "value":child_value - parent_value,
             "state_value":child_value
            }
        )
        node = node.parent
    return path[::-1]

def gather_paths(root:MCTSNode,selected_terminals: list[MCTSNode], pass_k: int,use_orm_reward:bool = False,use_chain_reward:bool=False,step_level_norm:bool=False,use_state_value_reward:bool=False,use_value_only:bool=False,average_one_generation:bool=False,advantage_mix_allancestor:bool=False) -> List[List[Dict[str, Any]]]:
    paths = []
    if len(selected_terminals) == 0:
        return paths
    if len(selected_terminals) != pass_k:
        pass_k = len(selected_terminals)
        # return None

    # 添加 selected_terminal 的叶子节点路径
    for terminal_node in selected_terminals:
        paths.append(path_from_root_to_node(terminal_node,average_one_generation))
    assert len(paths) == pass_k, f"Failed to generate {pass_k} paths,{len(paths)} instead"

    terminal_values = [leaf.accumulated_value for leaf in selected_terminals]
    if average_one_generation:
        root_value = root.accumulated_value
    else:
        root_value = root.accumulated_value / root.terminal_in_subtree

    if advantage_mix_allancestor:
        # print("use advantage mix all ancestor")
        for path in paths:
            # 每个节点的 value 都用其 state_value - 祖先.state_value，对所有祖先求平均
            for i in range(len(path)):
                if i == 0:
                    path[i]["value"] = path[i]["state_value"] - root_value
                else:
                    sum_value = 0
                    num_value = 0
                    for j in range(i):
                        sum_value += path[i]["state_value"] - path[j]["state_value"]
                        num_value += 1
                    sum_value += path[i]["state_value"] - root_value
                    num_value += 1
                    path[i]["value"] = sum_value / num_value
        return paths
    
    paths = fill_in_paths(paths)
    if use_chain_reward:
        # print("use chain reward in mcts!!")
        terminal_values = normalize_selected_terminals(selected_terminals)
        for path in paths:
            for node in path:
                node["value"] = terminal_values[paths.index(path)]
    elif use_orm_reward:
        # print("use orm reward in mcts!!")
        terminal_values = normalize_selected_terminals(selected_terminals)
        for path in paths:
            for node in path:
                # node["value"] = (node["value"] + terminal_values[paths.index(path)])/2
                node["value"] = (node["value"] + terminal_values[paths.index(path)])
    elif use_state_value_reward:
        # print("use state value reward in mcts!!")
        # paths = normalize_all_paths(paths,step_level_norm)
        for path in paths:
            for node in path:
                # node["value"] = (node["value"] + node["state_value"])/2
                node["value"] = (node["value"] + node["state_value"])
    elif use_value_only:
        # print("use value only in mcts!!")
        for path in paths:
            for node in path:
                node["value"] = node["state_value"]
    # else:
    #     print("use pure advantage in mcts!!")
    # print("path num",len(paths))
    return paths


def pass_rate(paths):
    if not paths:
        return 0
    pass_num = 0
    for path in paths:
        pass_num += path[-1]["pass_ratio"]
    pass_num /= len(paths)
    return pass_num


# 封装为一个函数,输入为item,输出为paths
def parallel_mcts(item, llm, tokenize_fn, detokenize_fn, args,system_prompt=None):
    return mcts_worker(item, llm, tokenize_fn, detokenize_fn, args["prompt_key"], args["answer_key"], args,system_prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    
    MODEL_PATH = "/data/o1-cloud/checkpoints/rl/qwen-14b-o1/epoch_1"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    # def tokenize_fn(texts, max_length=2048, device="cpu", system_prompt=None):
    #     sample_input_ids = tokenizer.encode(
    #         "[gMASK]<sop><|user|>\n" + texts[0][0],
    #         add_special_tokens=False
    #     )
    #     sample_input_ids = sample_input_ids[-max_length:] + \
    #         tokenizer.encode("<|assistant|>\n", add_special_tokens=False)
    #     return sample_input_ids

    import torch

    import json
    import torch
    def _tokenize_fn_llama(tokenizer, prompt, history, max_length,system_prompt=None):
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        if history:
            for x in history:
                conversation.append({"role": "user", "content": x["prompt"]})
                conversation.append({"role": "assistant", "content": x["response"]})
        conversation.append({"role": "user", "content": prompt})
        sample_input_ids = tokenizer.apply_chat_template(conversation)
        sample_input_ids = sample_input_ids[-max_length:] + tokenizer.encode("<|im_start|>assistant\n")
        # 把sample_input_ids decode成文本
        # with open("/workspace/lurui/openrlhf-glm/logs/outputs/sample_input_ids.jsonl", "a") as f:
        #     str_input_ids = tokenizer.decode(sample_input_ids)
        #     f.write(json.dumps({"sequences":str_input_ids,"system_prompt":system_prompt}) + "\n")
        return sample_input_ids


    def tokenize_fn(tokenizer, texts, max_length, device,system_prompt=None):
        batch = [_tokenize_fn_llama(tokenizer, prompt=_prompt, history=_history, max_length=max_length,system_prompt=system_prompt) for _prompt, _history in zip(*texts)]

        batch_length = max([len(x) for x in batch])
        pad_token_id = tokenizer.pad_token_id
        max_length = min(max_length, batch_length)

        def batch_encode_plus(input_ids):
            sample_len = len(input_ids)
            if sample_len < max_length:
                attention_mask = [0] * (max_length - sample_len) + [1] * sample_len
                input_ids = [pad_token_id] * (max_length - sample_len) + input_ids
            else:
                attention_mask = [1] * max_length
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            return {"input_ids": input_ids, "attention_mask": attention_mask}
        
        # batch = [batch_encode_plus(x) for x in batch]
        try:
            batch = tokenizer.batch_encode_plus(batch, return_tensors="pt", is_split_into_words=True, padding=True)
        except:
            batch = [batch_encode_plus(x) for x in batch]
            batch = {
                "input_ids": torch.stack([x["input_ids"] for x in batch]),
                "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            }
        return {k: v.to(device) for k, v in batch.items()}

    def decode_fn(ids):
        return tokenizer.decode(ids, skip_special_tokens=False)
    
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True
    )
    # args = {"temperature": 1.2, "top_p": 0.9, "max_depth": 40, "max_nodes": 256, "max_children": 4, "min_children":4, "shallow_enwide":False,"exploration_constant": 0.5, "prompt_key": "problem", "answer_key": "golden_answer", "backbone": "glm", "pass_k": 16, "backprop": 0, "max_node_per_depth": 18, "first_token_temperature": 0, "look_ahead": 0, "concurrent_num": 4, "path_num": 16,"prompt_max_len":1024,"max_token_num":4096,"max_time_use":6000,"step_level_norm":False,"random_pick":True,"use_orm_reward":False,"select_correct_leaf":False,"use_chain_reward":True,"use_state_value_reward":True,"use_pure_RM":True}
    args = {"temperature": 1.2, "top_p": 0.9, "max_depth": 40, "max_nodes": 512, "max_children": 4,"min_children":4, "shallow_enwide":False, "exploration_constant": 0.5, "prompt_key": "problem", "answer_key": "answer", "backbone": "qwen", "pass_k": 64, "backprop": 0, "max_node_per_depth": 64, "first_token_temperature": 1, "look_ahead": 0, "concurrent_num": 8, "path_num": 64,"prompt_max_len":1024,"max_token_num":8192,"max_time_use":6000,"step_level_norm":False,"random_pick":True,"use_orm_reward":False,"select_correct_leaf":True,"use_chain_reward":False,"use_state_value_reward":False,"use_value_only":True,"use_pure_RM":False,"use_pure_binary":True,"average_one_generation":True,"advantage_mix_allancestor":False}
    
    
    # input_file = "/data/share/glm-simple-evals/data/omni_math_500/test.jsonl"
    input_file = "/workspace/lurui/openrlhf-mcts/data/omni_math_500/part_" + str(parser.parse_args().index) + ".jsonl"

    # item = {"problem":data["Question"],"golden_answer":data["Answer"]}
    # item = {"problem":"Define\n\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]Find a way to write\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]in terms of $p$ and $q.$", "solution": "We count the number of times $\\frac{1}{n^3}$ appears in the sum\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3},\\]where $n$ is a fixed positive integer.  (In other words, we are conditioning the sum on $j + k$.)  We get a term of $\\frac{1}{n^3}$ each time $j + k = n.$  The pairs $(j,k)$ that work are $(1,n - 1),$ $(2,n - 2),$ $\\dots,$ $(n - 1,1),$ for a total of $n - 1$ pairs.  Therefore,\n\\begin{align*}\n\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3} &= \\sum_{n = 1}^\\infty \\frac{n - 1}{n^3} \\\\\n&= \\sum_{n = 1}^\\infty \\left( \\frac{n}{n^3} - \\frac{1}{n^3} \\right) \\\\\n&= \\sum_{n = 1}^\\infty \\left( \\frac{1}{n^2} - \\frac{1}{n^3} \\right) \\\\\n&= \\sum_{n = 1}^\\infty \\frac{1}{n^2} - \\sum_{n = 1}^\\infty \\frac{1}{n^3} \\\\\n&= \\boxed{p - q}.\n\\end{align*}", "golden_answer": "p - q"}
    # item = {"problem":"How many positive whole-number divisors does 196 have?", "solution": "First prime factorize $196=2^2\\cdot7^2$.  The prime factorization of any divisor of 196 cannot include any primes other than 2 and 7.  We are free to choose either 0, 1, or 2 as the exponent of 2 in the prime factorization of a divisor of 196.  Similarly, we may choose 0, 1, or 2 as the exponent of 7.  In total, there are $3\\times 3=9$ possibilities for the prime factorization of a divisor of 196.  Distinct prime factorizations correspond to distinct integers, so there are $\\boxed{9}$ divisors of 196.","golden_answer":"9"}
    # paths = parallel_mcts(item, llm, tokenize_fn,decode_fn, args)
    # with open("/workspace/lurui/openrlhf-mcts/data/finalpath.jsonl", "w") as f:
    #     json.dump(paths, f)

    with open(input_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
        for data in datas:
            item = {"problem":data["problem"],"golden_answer":data["answer"]}
            # item = {"problem":"Define\n\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]Find a way to write\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]in terms of $p$ and $q.$", "solution": "We count the number of times $\\frac{1}{n^3}$ appears in the sum\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3},\\]where $n$ is a fixed positive integer.  (In other words, we are conditioning the sum on $j + k$.)  We get a term of $\\frac{1}{n^3}$ each time $j + k = n.$  The pairs $(j,k)$ that work are $(1,n - 1),$ $(2,n - 2),$ $\\dots,$ $(n - 1,1),$ for a total of $n - 1$ pairs.  Therefore,\n\\begin{align*}\n\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3} &= \\sum_{n = 1}^\\infty \\frac{n - 1}{n^3} \\\\\n&= \\sum_{n = 1}^\\infty \\left( \\frac{n}{n^3} - \\frac{1}{n^3} \\right) \\\\\n&= \\sum_{n = 1}^\\infty \\left( \\frac{1}{n^2} - \\frac{1}{n^3} \\right) \\\\\n&= \\sum_{n = 1}^\\infty \\frac{1}{n^2} - \\sum_{n = 1}^\\infty \\frac{1}{n^3} \\\\\n&= \\boxed{p - q}.\n\\end{align*}", "golden_answer": "p - q"}
            # item = {"problem":"A group of $12$ pirates agree to divide a treasure chest of gold coins among themselves as follows. The $k^\\text{th}$ pirate to take a share takes $\\frac{k}{12}$ of the coins that remain in the chest. The number of coins initially in the chest is the smallest number for which this arrangement will allow each pirate to receive a positive whole number of coins. How many coins does the $12^{\\text{th}}$ pirate receive?\n$\\textbf{(A)} \\ 720 \\qquad  \\textbf{(B)} \\ 1296 \\qquad  \\textbf{(C)} \\ 1728 \\qquad  \\textbf{(D)} \\ 1925 \\qquad  \\textbf{(E)} \\ 3850$", "solution": "First prime factorize $196=2^2\\cdot7^2$.  The prime factorization of any divisor of 196 cannot include any primes other than 2 and 7.  We are free to choose either 0, 1, or 2 as the exponent of 2 in the prime factorization of a divisor of 196.  Similarly, we may choose 0, 1, or 2 as the exponent of 7.  In total, there are $3\\times 3=9$ possibilities for the prime factorization of a divisor of 196.  Distinct prime factorizations correspond to distinct integers, so there are $\\boxed{9}$ divisors of 196.","golden_answer":"9"}
            paths = parallel_mcts(item, llm, tokenize_fn, args)
            with open("/workspace/lurui/openrlhf-mcts/data/paths.json", "w") as f:
                json.dump(paths, f)