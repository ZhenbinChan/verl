from __future__ import annotations
import math
from typing import List, Optional, Callable
from pydantic import BaseModel
import json
import random
from collections import deque
from utils import *

prompt_dir = "mcts_utils/prompts"
with open(os.path.join(prompt_dir, "Z3ImplicationConversion1.txt"), "r") as f:
    system_prompt_z3_implication = f.read()

class MCTSNode(BaseModel):
    answer: str
    answer_token : List[int]
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    R : float = 0
    depth: int = 0
    main_chain: bool = False
    terminal: bool = False
    terminal_in_subtree: int = 0
    correct_terminal_in_subtree: int = 0
    selected_terminal_in_subtree: int = 0
    accumulated_value: float = 0
    repeat:bool = False
    value: float = 0
    finish_reason: Optional[str] = None
    reward_raw: float = None
    visits: int = 0
    heuristic: float = 0

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, value={self.value:.2f}, visits={self.visits})"


class TreeNode:
    def __init__(
        self,
        tree_idx: int,
        node_idx: int,
        decode_fn: Callable,
        token_id_list: List[int],
        log_prob_list: List[float],
        finish_reason: Optional[str] = None,
        is_end: bool = False,
        parent_node: Optional['TreeNode'] = None,
        parent_node_idx: Optional[int] = None,
        parent_node_split_idx: Optional[int] = None,
        child_nodes: Optional[List['TreeNode']] = None,
        child_split_indices: Optional[List[int]] = None,
        max_length: int = 7144,
        **kwargs
    ):
        """
        树节点的信息
        """
        # --- 分组信息 ---
        self.tree_idx: int = tree_idx  # 树的索引
        self.node_idx: int = node_idx  # 节点的索引

        # --- 节点包含的文本信息 ---
        self.token_id_list: List[int] = token_id_list
        self.token_str_list: List[str] = [
            decode_fn([token_id]) for token_id in token_id_list]

        self.log_prob_list: List[float] = log_prob_list
        self.token_num: int = len(token_id_list)  # token 数量
        self.finish_reason: Optional[str] = finish_reason  # 结束原因
        self.is_end: bool = is_end  # 是否是叶子节点

        # --- 父亲节点信息 ---
        self.parent_node = parent_node  # 父节点对象
        self.parent_node_idx = parent_node_idx  # 父节点的索引
        self.parent_node_split_idx = parent_node_split_idx  # 从父节点分叉的 token 索引

        # --- 孩子节点信息 ---
        # 子节点列表
        self.child_nodes: List['TreeNode'] = child_nodes if child_nodes else []
        # 子节点分叉的 token 索引
        self.child_split_indices: List[int] = child_split_indices if child_split_indices else [
        ]

        # --- 孩子正确率信息（分段） ---
        self.child_correct_num: List[int] = []
        self.child_total_num: List[int] = []

        # --- 截止到目前的 aggregate 字符串以及所有完整字符串（减少遍历时间，以及用于判断答案） ---
        self.answer: str = ''.join(self.token_str_list)
        self.aggregate_str: str = ""
        if parent_node is not None:
            parent_token_str_list = parent_node.token_str_list
            self.aggregate_str = parent_node.aggregate_str + \
                ''.join(parent_token_str_list[:parent_node_split_idx])
        self.total_str: str = self.aggregate_str + self.answer

        self.aggregate_token_ids: List[int] = []
        if parent_node is not None:
            self.aggregate_token_ids = parent_node.aggregate_token_ids + \
                parent_node.token_id_list[:parent_node_split_idx]

        # --- 掩码信息 ---
        self.mask: List[bool] = [False] * len(self.token_str_list)
        
        # 如果是旁枝，第一个 token 就不需要再被选中了
        if len(self.aggregate_token_ids) > 0 and len(self.token_str_list) > 0:
            self.mask[0] = True
        
        # 检查是否超过最大长度
        total_length = len(self.aggregate_token_ids) + len(self.token_id_list)
        if total_length > max_length:
            # 计算需要 mask 的 token 数量
            tokens_to_mask = total_length - max_length
            # 从后往前 mask 掉超出长度的 tokens
            for i in range(max(0, len(self.mask) - tokens_to_mask), len(self.mask)):
                self.mask[i] = True
            self.is_end = True
        
        # 检查特殊token
        for i, token_str in enumerate(self.token_str_list):
            if "conclusion" in token_str.lower() or "answer" in token_str.lower():
                # 掩盖后续 tokens
                for j in range(i + 1, len(self.mask)):
                    self.mask[j] = True
                self.is_end = True
                break

        # --- 节点的分数 ---
        self.binary_score: Optional[float] = None
        self.score: Optional[float] = None

        # --- 节点的划分 ---
        self.segments = [0] + [i + 1 for i, token_str in enumerate(self.token_str_list) if '\n\n' in token_str] + [len(self.token_str_list)]
        # if self.segments[-1] != len(self.token_str_list):
        #     self.segments.append(len(self.token_str_list))
        self.token_id2segment_id = [0] * len(self.token_id_list)
        for i in range(len(self.segments) - 1):
            start, end = self.segments[i], self.segments[i + 1]
            for j in range(start, end):
                self.token_id2segment_id[j] = i
        self.heuristic = 0

        # --- 节点的蕴含信息（分段） ---
        self.evaluation_strategy = kwargs.get('evaluation_strategy', 'token-entropy')
        self.token_score = []
        self.segment_scores = []
        self.implication = []
        self.aggregate_implication = kwargs.get('aggregate_implication', [])
        self.compute_token_score(**kwargs)

        self.aggregate_segment_scores = []
        if parent_node is not None:
            self.aggregate_segment_scores = parent_node.aggregate_segment_scores + \
                parent_node.segment_scores[:parent_node.token_id2segment_id[parent_node_split_idx]]
        self.all_segment_scores = self.aggregate_segment_scores + self.segment_scores
        self.heuristic = sum(self.all_segment_scores) / len(self.all_segment_scores)

    def get_prefix(self, current_token_index: int) -> str:
        """
        给定截断的位置，获取前缀文本，通过迭代构建，从根节点到当前节点。

        :return: 拼接后的前缀字符串。
        """
        parent_tokens = self.aggregate_str
        return parent_tokens + ''.join(self.token_str_list[:current_token_index])

    def get_prefix_ids(self, current_token_index: int) -> List[int]:
        """
        给定截断的位置，获取前缀 token_ids

        :return: 拼接后的前缀 token_ids 列表。
        """

        parent_token_ids = self.aggregate_token_ids
        return parent_token_ids + self.token_id_list[:current_token_index]
    
    def get_prefix_implication(self, current_token_index: int) -> str:
        return self.aggregate_implication + self.implication[:self.token_id2segment_id[current_token_index]]

    def add_child(self, child_node: 'TreeNode', split_index: int) -> None:
        """
        添加子节点。

        :param child_node: TreeNode 对象。
        :param split_index: 分裂的token索引。
        """
        self.child_nodes.append(child_node)
        self.child_split_indices.append(split_index)
        child_node.parent_node = self

    def compute_token_score(self, **kwargs) -> None:
        if self.evaluation_strategy == 'token-entropy':
            self.token_score = self.log_prob_list
            self.segment_scores = self.log_prob_list
            return
        
        label2score = {
            "ENTAILMENT": 1,
            "NEUTRAL": 0,
            "CONTRADICTION": -1,
            "UNKNOWN": 0,
            "ERROR": 0
        }

        self.token_score = [0] * len(self.token_str_list)
        self.segment_scores = []

        if self.evaluation_strategy == 'segment-entropy':
            for start, end in zip(self.segments[:-1], self.segments[1:]):
                log_probs = self.log_prob_list[start:end]
                self.segment_scores.append(sum(log_probs) / len(log_probs))

        elif self.evaluation_strategy in 'nli':
            parsed_chain = parse_reasoning_steps(self.answer)
            results_nli = verify_steps_nli(parsed_chain)
            self.segment_scores = [label2score[result['label']] * result['score'] for result in results_nli]

        elif self.evaluation_strategy == 'fol':
            usr_input = f"Question:\n{kwargs['question']}\n\n" \
                + f"Reasoning steps:\n{self.total_str}\n\n" \
                + f"Z3 Declaration:\n{kwargs['declaration']}"
            try:
                prefix_implication = construct_implication_prefix(self.aggregate_implication)
                implication = get_response(usr_input, system_prompt_z3_implication, prefix_implication)
                parsed_implication = parse_python_logic_steps(extract_python_code(implication))
                results_fol = verify_steps_fol(kwargs['declaration'], parsed_implication)
                self.implication = parsed_implication
                self.segment_scores = [label2score[result['label']] * 1.0 for result in results_fol]
            except Exception as e:
                pass

        if len(self.segment_scores) == 0:
            self.segment_scores = [0] * (len(self.segments) - 1)

        for start, end, score in zip(self.segments[:-1], self.segments[1:], self.segment_scores):
            self.token_score[start] = score
            self.mask[start+1:end] = [True] * (end - start - 1)
        self.mask[end:] = [True] * (len(self.token_str_list) - end)


    def get_max_entropy_tokens(self, top_n: int = 1) -> List[int]:
        """
        获取最高熵的token索引，返回top_n个。
        只考虑未被 mask 的 token。

        :param top_n: 需要返回的最高熵token数量。
        :return: 最高熵token的索引列表。
        """
        entropies = []
        for i, score in enumerate(self.token_score):
            if not self.mask[i]:  # 只考虑未被 mask 的 token
                entropy = -score  # 简单地用负对数概率作为熵
                entropies.append((entropy, i))

        # 按熵值排序并返回前 top_n 个索引
        sorted_indices = sorted(entropies, key=lambda x: x[0], reverse=True)
        result = [idx for _, idx in sorted_indices[:top_n]]

        # 如果不够 top_n 个，复制若干份，确保鲁棒性
        # 但这样可能会导致重复，所以目前不使用
        # while len(result) < top_n:
        #     result += result[:top_n - len(result)]

        return result

def build_into_tree_format(
    tree_lists,
    decode_fn,
    num_traces,balance_ratio=0,
    average_one_generation=False,
    use_weighted_value=False,
    use_all_terminals=False,
    weighted_value_style="uniform",
    overall_norm_style = "token",
    inner_repetition_penalty=False) -> MCTSNode:
    all_leaves = []
    try:
        def convert_to_json(node: MCTSNode):
            if not node.children:
                return {
                    "answer": node.answer,
                    "answer_token": node.answer_token,
                    "R" : node.R,
                    "depth": node.depth,
                    "main_chain": node.main_chain,
                    "terminal": node.terminal,
                    "terminal_in_subtree": node.terminal_in_subtree,
                    "correct_terminal_in_subtree": node.correct_terminal_in_subtree,
                    "selected_terminal_in_subtree": node.selected_terminal_in_subtree,
                    "accumulated_value": node.accumulated_value,
                    "finish_reason": node.finish_reason,
                }
            else:
                return {
                    "answer": node.answer,
                    "answer_token": node.answer_token,
                    "R" : node.R,
                    "depth": node.depth,
                    "main_chain": node.main_chain,
                    "terminal": node.terminal,
                    "children": [convert_to_json(child) for child in node.children],
                    "terminal_in_subtree": node.terminal_in_subtree,
                    "correct_terminal_in_subtree": node.correct_terminal_in_subtree,
                    "selected_terminal_in_subtree": node.selected_terminal_in_subtree,
                    "accumulated_value": node.accumulated_value,
                    "finish_reason": node.finish_reason,
                }
                
        def build_tree_node(decode_fn,tree_node: TreeNode, parent_mcts_node: Optional[MCTSNode] = None) -> MCTSNode:
            # 对 child_nodes 按照 parent_node_split_idx 进行排序
            tree_node.child_nodes.sort(key=lambda x: x.parent_node_split_idx)

            # 存储所有孩子节点的 parent_node_split_idx
            child_split_indices = [child.parent_node_split_idx for child in tree_node.child_nodes]
            
            is_terminal = False
            R = 0
            heuristic = 0
            main_chain = False
            if not child_split_indices:
                # 如果没有孩子节点，设置end_idx为整个token_id_list的长度
                first_child_split_idx = len(tree_node.token_id_list)
                is_terminal = True
                R = tree_node.score
                heuristic = tree_node.heuristic
                if tree_node.binary_score == 1:
                    main_chain = True
            else:
                first_child_split_idx = child_split_indices[0]

            # 初始节点段，从0到第一个孩子的分割位置
            root_node = MCTSNode(
                answer=''.join([decode_fn([token_id]) for token_id in tree_node.token_id_list[:first_child_split_idx]]),
                answer_token=tree_node.token_id_list[:first_child_split_idx],
                parent=parent_mcts_node,
                depth=(parent_mcts_node.depth + 1) if parent_mcts_node else 0,
                terminal=is_terminal,
                R = R,
                main_chain = main_chain,
                finish_reason=tree_node.finish_reason,
                heuristic=heuristic,
            )
            
            if root_node.terminal:
                all_leaves.append(root_node)

            # 递归构建子树
            def add_segments_and_children(current_mcts_node: MCTSNode, start_idx: int):
                i = 0
                while i < len(tree_node.child_nodes):
                    child_nodes_group = []
                    current_split_idx = child_split_indices[i]
                    
                    # 收集所有具有相同 parent_node_split_idx 的孩子节点
                    while i < len(tree_node.child_nodes) and child_split_indices[i] == current_split_idx:
                        child_nodes_group.append(tree_node.child_nodes[i])
                        i += 1
                    is_terminal = False
                    R = 0
                    heuristic = 0
                    main_chain = False
                    if i < len(tree_node.child_nodes):
                        next_split_idx = child_split_indices[i]
                    else:
                        next_split_idx = len(tree_node.token_id_list)
                        is_terminal = True
                        R = tree_node.score
                        heuristic = tree_node.heuristic
                        if tree_node.binary_score == 1:
                            main_chain = True
                    
                    # 创建当前段后的子段
                    segment_node = MCTSNode(
                        answer=''.join([decode_fn([token_id]) for token_id in tree_node.token_id_list[start_idx:next_split_idx]]),
                        answer_token=tree_node.token_id_list[start_idx:next_split_idx],
                        parent=current_mcts_node,
                        depth=current_mcts_node.depth + 1,
                        terminal=is_terminal,
                        R = R,
                        main_chain = main_chain,
                        finish_reason=tree_node.finish_reason,
                        heuristic=heuristic,
                    )
                    current_mcts_node.children.append(segment_node)
                    if segment_node.terminal:
                        all_leaves.append(segment_node)
                    
                    # 为每一个子节点组添加子树, 并将子树挂载到segment_node
                    for child_node in child_nodes_group:
                        child_mcts_node = build_tree_node(decode_fn, child_node, current_mcts_node)
                        current_mcts_node.children.append(child_mcts_node)
                    
                    start_idx = next_split_idx
                    # 更新当前父节点
                    current_mcts_node = segment_node

            # 初始调用，为根节点添加子段
            add_segments_and_children(root_node, first_child_split_idx)
            
            return root_node

        # 构建（虚拟）根节点
        root = MCTSNode(
            answer="",
            answer_token=[]
        )
        
        # 根的所有孩子是所有tree_lists[i][0]
        for i, tree_list in enumerate(tree_lists):
            if len(tree_list) > 0:
                root.children.append(build_tree_node(decode_fn, tree_list[0], root))
        
        leaf_normalize(all_leaves, root, average_one_generation, overall_norm_style, inner_repetition_penalty)
        if use_all_terminals:
            selected_terminals = all_leaves
        else:
            selected_terminals = select_terminal(all_leaves, num_traces, balance_ratio)

        if use_weighted_value:
            print("weighted value")
            for leaf in selected_terminals:
                selected_backpropagate(leaf)
            compute_weighted_update(root, reweighted_value_style=weighted_value_style)
        return root, selected_terminals
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        from IPython import embed
        embed()
      
      
def leaf_normalize(nodes,root,average_one_generation:bool = False,overall_norm_style = "token",inner_repetition_penalty:bool = False):
    leaf_correctness = [leaf.R for leaf in nodes]
    # print("leaf_correctness",leaf_correctness)
    _sum = sum(leaf_correctness)
    num = len(leaf_correctness) - 1
    if num == 0:
        # return
        assert False, "entropy num_traces == 0"
    else:
        mean = [(_sum - leaf_correctness[i]) / num for i in range(len(leaf_correctness))]
        # root.reward_raw用所有的leaf_correctness的平均值
        root.reward_raw = sum(leaf_correctness) / len(leaf_correctness)
        for i, leaf in enumerate(nodes):
            leaf.R = leaf.R - mean[i]
            leaf.accumulated_value = leaf.R
            if inner_repetition_penalty:
                if leaf.finish_reason != "stop":
                    leaf.R = -1
                    leaf.accumulated_value = leaf.R
            leaf_backpropagate(leaf)
        if average_one_generation:
            compute_accumulated_value(root)
    normalize_all_steps(root, overall_norm_style)
            
def leaf_backpropagate(node: MCTSNode):
    if node.terminal and node.main_chain:
        node.terminal_in_subtree += 1 # 子树中的终止节点数
        node.correct_terminal_in_subtree += 1 # 正确路径数
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

def normalize_all_steps(root: MCTSNode, overall_norm_style: str = "token"):
    # 从root开始遍历所有节点，对所有terminal_in_subtree！=0节点的accumulated_value进行归一化
    all_steps = []
    to_consider = deque([root])
    while to_consider:
        current_node = to_consider.popleft()
        if current_node.terminal_in_subtree != 0 or current_node.terminal:
            all_steps.append(current_node)
        to_consider.extend(current_node.children)

    # print("all_step value",len(all_steps))
    if overall_norm_style == "step":
        print("step level normalization")
        step_sum = 0
        step_num = 0
        for node in all_steps:
            # step_sum += node.accumulated_value*node.terminal_in_subtree
            # step_num += node.terminal_in_subtree
            step_sum += node.accumulated_value
            step_num += node.terminal_in_subtree
        if step_num == 0:
            mean = 0
        else:
            mean = step_sum/step_num
        print("mean:", mean,step_sum,step_num)
        for node in all_steps:
            node.accumulated_value = node.accumulated_value - mean * node.terminal_in_subtree
    elif overall_norm_style == "token":
        # print("token level normalization")
        step_sum = 0
        step_num = 0
        for node in all_steps:
            # step_sum += node.accumulated_value*node.terminal_in_subtree*len(node.answer_token)
            # step_num += node.terminal_in_subtree*len(node.answer_token)
            step_sum += node.accumulated_value*len(node.answer_token)
            step_num += node.terminal_in_subtree*len(node.answer_token)
        if step_num == 0:
            mean = 0
        else:
            mean = step_sum/step_num
        # print("mean:", mean)
        for node in all_steps:
            node.accumulated_value = node.accumulated_value - mean * node.terminal_in_subtree
    else:
        return
        
def compute_accumulated_value(node: MCTSNode):
    if not node.children:  # If the node is a leaf node
        return node.accumulated_value

    # Post-order traversal: first process all children
    total_value = 0
    terminal_children = 0
    for child in node.children:
        if child.terminal_in_subtree > 0:
            terminal_children += 1
            total_value += compute_accumulated_value(child)

    # Calculate the average accumulated value for the current node
    node.accumulated_value = total_value / terminal_children if terminal_children > 0 else 0
    return node.accumulated_value

def select_terminal(nodes: List[MCTSNode], num_traces: int, balance_ratio: float = 0):
    if balance_ratio < 0:
        nodes.sort(key=lambda x: x.heuristic, reverse=True)
        return nodes[:num_traces]
    
    elif balance_ratio == 0:
        random.shuffle(nodes)
        selected_terminals = []
        remaining_terminals = []

        # Traverse the shuffled paths and select the first pass_ratio == 1 path
        for node in nodes:
            if node.main_chain and len(selected_terminals) == 0:
                selected_terminals.append(node)
            else:
                remaining_terminals.append(node)

        # Calculate how many additional paths we need
        remaining_num_traces = num_traces - len(selected_terminals)

        # Randomly select remaining_num_traces paths from the remaining_paths if possible
        if remaining_num_traces > 0:
            selected_terminals.extend(random.sample(remaining_terminals, min(
                remaining_num_traces, len(remaining_terminals))))

        # Shuffle the selected paths to ensure they are returned in random order
        random.shuffle(selected_terminals)
        assert len(
            selected_terminals) == num_traces, f"len(selected_paths) = {len(selected_terminals)} != num_traces = {num_traces}"

        return selected_terminals
    else:
        random.shuffle(nodes)
        num_correct_needed = int(num_traces * balance_ratio)
        num_incorrect_needed = int(num_traces * balance_ratio)

        selected_correct = []
        selected_incorrect = []
        remaining_terminals = []

        # Traverse the shuffled nodes and select correct and incorrect nodes
        for node in nodes:
            if node.main_chain and len(selected_correct) < num_correct_needed:
                selected_correct.append(node)
            elif not node.main_chain and len(selected_incorrect) < num_incorrect_needed:
                selected_incorrect.append(node)
            else:
                remaining_terminals.append(node)
        
        # Calculate how many additional terminals we need
        num_selected = len(selected_correct) + len(selected_incorrect)
        remaining_num_traces = num_traces - num_selected
        # print(f"num_correct = {len(selected_correct)}, num_incorrect = {len(selected_incorrect)},remaining_num_traces = {remaining_num_traces}")

        selected_terminals = selected_correct + selected_incorrect

        # Randomly select remaining_num_traces paths from remaining_terminals if possible
        if remaining_num_traces > 0:
            selected_terminals.extend(random.sample(remaining_terminals, min(remaining_num_traces, len(remaining_terminals))))

        # Shuffle the selected terminals to ensure they are returned in random order
        random.shuffle(selected_terminals)
        assert len(selected_terminals) == num_traces, f"len(selected_terminals) = {len(selected_terminals)} != num_traces = {num_traces}"

        return selected_terminals

def selected_backpropagate(node: MCTSNode):
    node.selected_terminal_in_subtree += 1
    # 所有父亲的terminal_in_subtree都加1
    parent = node.parent
    while parent:
        parent.selected_terminal_in_subtree += 1
        parent = parent.parent

def compute_weighted_update(node: MCTSNode, reweighted_value_style="original"):
    if node.selected_terminal_in_subtree == 0:
        return
    # assert reweighted_value_style == "sqrt"
    if reweighted_value_style == "sqrt":
        node.accumulated_value = node.accumulated_value / math.sqrt(node.selected_terminal_in_subtree)
    elif reweighted_value_style == "uniform":
        node.accumulated_value = node.accumulated_value / node.selected_terminal_in_subtree
    elif reweighted_value_style == "original":
        node.accumulated_value = node.accumulated_value
    else:
        raise ValueError(f"Unsupported rewegihted_value_style: {reweighted_value_style}")
    
    # node.accumulated_value = node.accumulated_value * node.terminal_in_subtree / node.selected_terminal_in_subtree
    # node.accumulated_value = node.accumulated_value * node.terminal_in_subtree / node.selected_terminal_in_subtree
    
    for child in node.children:
        compute_weighted_update(child, reweighted_value_style=reweighted_value_style)