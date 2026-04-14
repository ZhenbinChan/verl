"""Entropy-chain reward manager.

Computes per-leaf rewards using tree-level value back-propagation, then
applies the formula:

    G_A(s) = V(s) - V(root)
    L_A(s) = V(s) - V(parent(s))
    R(s)   = G_A(s) + L_A(s)

The value back-propagation logic mirrors
``mcts_utils/tree_node.py`` (leaf_normalize, leaf_backpropagate,
normalize_all_steps) but is re-implemented here to avoid importing
the full ``mcts_utils`` module tree (which has heavy side-effects such
as loading an NLI classifier at import time).
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


# ---------------------------------------------------------------------------
# Deserialization of tree metadata (plain dicts -> _TreeProxy objects)
# ---------------------------------------------------------------------------

class _TreeProxy:
    """Lightweight stand-in for ``TreeNode`` reconstructed from serialized dicts.

    Provides the same attribute interface consumed by ``_build_value_tree``.
    """

    __slots__ = (
        "node_idx", "token_id_list", "score", "binary_score",
        "finish_reason", "is_end", "parent_node_split_idx",
        "child_nodes", "_child_node_indices",
    )

    def __init__(self, data: dict):
        self.node_idx: int = data["node_idx"]
        self.token_id_list: List[int] = data["token_id_list"]
        self.score = data["score"]
        self.binary_score = data["binary_score"]
        self.finish_reason = data["finish_reason"]
        self.is_end: bool = data["is_end"]
        self.parent_node_split_idx = data.get("parent_node_split_idx")
        self.child_nodes: List[_TreeProxy] = []
        self._child_node_indices: List[int] = data.get("child_node_indices", [])


def _deserialize_tree_lists(serialized: List[List[dict]]) -> List[List[_TreeProxy]]:
    """Reconstruct ``_TreeProxy`` trees from the plain-dict format produced by
    ``_serialize_tree_lists`` in ``entropy_chain_expander.py``."""
    result: List[List[_TreeProxy]] = []
    for tree_data in serialized:
        nodes = [_TreeProxy(d) for d in tree_data]
        idx_map = {n.node_idx: n for n in nodes}
        for proxy in nodes:
            proxy.child_nodes = [idx_map[ci] for ci in proxy._child_node_indices]
        result.append(nodes)
    return result


# ---------------------------------------------------------------------------
# Lightweight value node (mirrors MCTSNode for value tracking only)
# ---------------------------------------------------------------------------

class _ValueNode:
    """Minimal tree node used exclusively for value back-propagation."""

    __slots__ = (
        "answer_token",
        "parent",
        "children",
        "R",
        "terminal",
        "main_chain",
        "finish_reason",
        "accumulated_value",
        "terminal_in_subtree",
        "correct_terminal_in_subtree",
        "reward_raw",
        "source_tree_idx",
        "source_node_idx",
    )

    def __init__(
        self,
        answer_token: List[int],
        parent: Optional["_ValueNode"] = None,
        terminal: bool = False,
        R: float = 0.0,
        main_chain: bool = False,
        finish_reason: Optional[str] = None,
        source_tree_idx: int = -1,
        source_node_idx: int = -1,
    ):
        self.answer_token = answer_token
        self.parent = parent
        self.children: List[_ValueNode] = []
        self.R = R
        self.terminal = terminal
        self.main_chain = main_chain
        self.finish_reason = finish_reason
        self.accumulated_value: float = 0.0
        self.terminal_in_subtree: int = 0
        self.correct_terminal_in_subtree: int = 0
        self.reward_raw: Optional[float] = None
        self.source_tree_idx = source_tree_idx
        self.source_node_idx = source_node_idx


# ---------------------------------------------------------------------------
# Tree construction (mirrors build_into_tree_format in tree_node.py)
# ---------------------------------------------------------------------------

def _build_value_tree(
    tree_lists,
    overall_norm_style: str = "token",
    inner_repetition_penalty: bool = False,
) -> Tuple[_ValueNode, List[_ValueNode]]:
    """Convert ``TreeNode`` lists into a ``_ValueNode`` tree and run value
    back-propagation (leaf_normalize + leaf_backpropagate + normalize_all_steps).

    Returns ``(virtual_root, all_leaves)``.

    - 将多条推理树（每条是扁平的节点列表，根在 ``tree_list[0]``）挂到一个「虚拟根」下，
      建成仅用于估值回传的 ``_ValueNode`` 树（结构与 ``mcts_utils/tree_node.py`` 中
      ``build_into_tree_format`` 一致，避免直接依赖 mcts_utils）。
    - 叶节点：原树中不再有子节点的节点；叶上的 ``R`` 来自原 ``score``，非叶段 ``R`` 为 0。
    - 多叶时调用 ``_leaf_normalize`` 做留一中心化并向上回传；单叶时直接把根 ``reward_raw``
      设为该叶分数。
    """

    all_leaves: List[_ValueNode] = []

    def _build_one(tree_node, parent_vnode: Optional[_ValueNode], tree_idx: int) -> _ValueNode:
        # 同一父边上的分叉按「在父 token 序列上的切分位置」排序，便于按序列从左到右建段。
        tree_node.child_nodes.sort(key=lambda x: x.parent_node_split_idx)
        child_split_indices = [c.parent_node_split_idx for c in tree_node.child_nodes]

        # 无子节点：整段 token 是一条走到头的路径，用原节点的 score / binary_score 作为终端信息。
        is_terminal = not child_split_indices
        R = float(tree_node.score or 0.0) if is_terminal else 0.0
        main_chain = bool(getattr(tree_node, "binary_score", None) == 1) if is_terminal else False
        # 第一个 _ValueNode 覆盖的 token 范围：[0, first_split)；有子则 first_split 为首个分叉位置。
        first_split = len(tree_node.token_id_list) if is_terminal else child_split_indices[0]

        root_vnode = _ValueNode(
            answer_token=tree_node.token_id_list[:first_split],
            parent=parent_vnode,
            terminal=is_terminal,
            R=R,
            main_chain=main_chain,
            finish_reason=tree_node.finish_reason,
            source_tree_idx=tree_idx,
            source_node_idx=tree_node.node_idx,
        )
        if root_vnode.terminal:
            all_leaves.append(root_vnode)

        # 有子节点时：在「根段」之后沿 token_id_list 继续切分，插入中间段与子树。
        if child_split_indices:
            _add_segments_and_children(
                tree_node, root_vnode, child_split_indices, first_split, tree_idx,
            )
        return root_vnode

    def _add_segments_and_children(
        tree_node,
        current_vnode: _ValueNode,
        child_split_indices: List[int],
        start_idx: int,
        tree_idx: int,
    ):
        # 按「分叉位置」分组：同一 parent_node_split_idx 的多个子节点是同一切点上的并列分支。
        i = 0
        while i < len(tree_node.child_nodes):
            current_split_idx = child_split_indices[i]
            child_group = []
            while i < len(tree_node.child_nodes) and child_split_indices[i] == current_split_idx:
                child_group.append(tree_node.child_nodes[i])
                i += 1

            # 若后面没有更多子分组，下一段一直延伸到当前 TreeNode 的 token 末尾：该段为终端段，携带本节点 score。
            # 否则下一段只延伸到「下一组分叉」位置，中间段非终端、R=0。
            is_terminal = i >= len(tree_node.child_nodes)
            if is_terminal:
                next_split_idx = len(tree_node.token_id_list)
                R = float(tree_node.score or 0.0)
                main_chain = bool(getattr(tree_node, "binary_score", None) == 1)
            else:
                next_split_idx = child_split_indices[i]
                R = 0.0
                main_chain = False

            seg = _ValueNode(
                answer_token=tree_node.token_id_list[start_idx:next_split_idx],
                parent=current_vnode,
                terminal=is_terminal,
                R=R,
                main_chain=main_chain,
                finish_reason=tree_node.finish_reason,
                source_tree_idx=tree_idx,
                source_node_idx=tree_node.node_idx,
            )
            # 先挂「当前切点到下一分叉/结尾」的 token 段，再挂该切点上的各子树（递归 _build_one）。
            current_vnode.children.append(seg)
            if seg.terminal:
                all_leaves.append(seg)

            for child_node in child_group:
                child_vnode = _build_one(child_node, current_vnode, tree_idx)
                current_vnode.children.append(child_vnode)

            # 继续沿同一 TreeNode 的 token 序列向右推进。
            start_idx = next_split_idx
            current_vnode = seg

    # 虚拟根：无 token，用于把多条样本树（tree_lists 的每个元素）接在同一父节点下。
    virtual_root = _ValueNode(answer_token=[])
    for tree_idx, tree_list in enumerate(tree_lists):
        if tree_list:
            child = _build_one(tree_list[0], virtual_root, tree_idx)
            virtual_root.children.append(child)

    # 多叶：留一中心化 + 回传 + 全局归一（见 _leaf_normalize / 后续步骤）。
    if len(all_leaves) > 1:
        _leaf_normalize(all_leaves, virtual_root, overall_norm_style, inner_repetition_penalty)
    # 单叶：无相对比较，根上直接记下原始回报，叶上累计值置 0。
    elif len(all_leaves) == 1:
        virtual_root.reward_raw = all_leaves[0].R
        all_leaves[0].accumulated_value = 0.0
        all_leaves[0].terminal_in_subtree = 1
        virtual_root.terminal_in_subtree = 1

    return virtual_root, all_leaves


# ---------------------------------------------------------------------------
# Value back-propagation helpers (mirrors tree_node.py)
# ---------------------------------------------------------------------------

def _leaf_normalize(
    leaves: List[_ValueNode],
    root: _ValueNode,
    overall_norm_style: str,
    inner_repetition_penalty: bool,
):
    """Leave-one-out centering → back-propagate → global normalize."""
    correctness = [leaf.R for leaf in leaves]
    total = sum(correctness)
    n = len(correctness)
    root.reward_raw = total / n

    for i, leaf in enumerate(leaves):
        mean_i = (total - correctness[i]) / (n - 1)
        leaf.R = correctness[i] - mean_i
        leaf.accumulated_value = leaf.R
        if inner_repetition_penalty and leaf.finish_reason != "stop":
            leaf.R = -1.0
            leaf.accumulated_value = leaf.R
        _leaf_backpropagate(leaf)

    _normalize_all_steps(root, overall_norm_style)


def _leaf_backpropagate(node: _ValueNode):
    node.terminal_in_subtree += 1
    if node.main_chain:
        node.correct_terminal_in_subtree += 1

    parent = node.parent
    while parent is not None:
        parent.terminal_in_subtree += 1
        if node.main_chain:
            parent.correct_terminal_in_subtree += 1
        parent.accumulated_value += node.accumulated_value
        parent = parent.parent


def _normalize_all_steps(root: _ValueNode, style: str):
    if style not in ("step", "token"):
        return

    queue: deque[_ValueNode] = deque([root])
    all_steps: List[_ValueNode] = []
    while queue:
        cur = queue.popleft()
        if cur.terminal_in_subtree != 0 or cur.terminal:
            all_steps.append(cur)
        queue.extend(cur.children)

    step_sum = 0.0
    step_num = 0.0
    for nd in all_steps:
        tok_len = max(len(nd.answer_token), 1)
        weight = tok_len if style == "token" else 1
        step_sum += nd.accumulated_value * weight
        step_num += nd.terminal_in_subtree * weight

    mean = step_sum / step_num if step_num != 0 else 0.0
    for nd in all_steps:
        nd.accumulated_value -= mean * nd.terminal_in_subtree


def _node_value(node: _ValueNode) -> float:
    if node.terminal_in_subtree == 0:
        return 0.0
    return node.accumulated_value / node.terminal_in_subtree


def _value_nodes_root_to_leaf(leaf: _ValueNode) -> List[_ValueNode]:
    """Segments on the path from virtual root to ``leaf`` (excluding virtual root)."""
    chain: List[_ValueNode] = []
    node = leaf
    while node.parent is not None:
        chain.append(node)
        node = node.parent
    return chain[::-1]


def _path_from_root_to_leaf(leaf: _ValueNode) -> List[Dict[str, Any]]:
    """TreeRL-style path extraction from a leaf to root."""
    path: List[Dict[str, Any]] = []
    node = leaf
    while node.parent is not None:
        parent_value = _node_value(node.parent)
        child_value = _node_value(node)
        path.append(
            {
                "token_answer": node.answer_token,
                "value": child_value - parent_value,
                "state_value": child_value,
            }
        )
        node = node.parent
    return path[::-1]


def _fill_in_paths(path: List[Dict[str, Any]], epsilon: float = 1e-8) -> List[Dict[str, Any]]:
    """Mirror TreeRL fill_in_paths: propagate previous non-zero value."""
    for i in range(1, len(path)):
        if abs(path[i]["value"]) < epsilon:
            path[i]["value"] = path[i - 1]["value"]
    return path


def _apply_reward_style(path: List[Dict[str, Any]], reward_style: str, root_value: float) -> List[Dict[str, Any]]:
    """Apply TreeRL-compatible value mixing strategy on a path."""
    if reward_style == "state_value":
        for node in path:
            node["value"] = node["value"] + node["state_value"] # 相当于 V(s_n) + V(s_n) - V(parent(s_n))
    elif reward_style == "value_only":
        for node in path:
            node["value"] = node["state_value"]
    elif reward_style == "pure_advantage":
        pass
    elif reward_style == "ga_plus_la":
        for node in path:
            node["value"] = (node["state_value"] - root_value) + node["value"]
    else:
        raise ValueError(f"Unsupported reward_style: {reward_style}")
    return path


# ---------------------------------------------------------------------------
# EntropyRewardManager
# ---------------------------------------------------------------------------

class EntropyRewardManager:
    """Reward manager for the *treerl* (entropy-chain) sampling strategy.

    Tree-level values are back-propagated first, then each root->leaf path is
    converted into dense token-level rewards. By default (`reward_style` set to
    `"state_value"`), it follows TreeRL's `use_state_value_reward` behavior:

        value_step = (V_child - V_parent) + V_child

    and broadcasts each step value to all tokens in the corresponding segment.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        overall_norm_style: str = "token",
        inner_repetition_penalty: bool = False,
        reward_style: str = "state_value",
        print_entropy_tree: bool = False,
        print_entropy_tree_max_preview_chars: int = 80,
        print_entropy_tree_local_rank_only: bool = True,
        entropy_tree_graphviz_dir: Optional[str] = None,
        entropy_tree_graphviz_view: bool = False,
        entropy_tree_graphviz_max_label_chars: int = 48,
        entropy_tree_graphviz_local_rank_only: bool = True,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overall_norm_style = overall_norm_style
        self.inner_repetition_penalty = inner_repetition_penalty
        self.reward_style = reward_style
        self.print_entropy_tree = print_entropy_tree
        self.print_entropy_tree_max_preview_chars = print_entropy_tree_max_preview_chars
        self.print_entropy_tree_local_rank_only = print_entropy_tree_local_rank_only
        self.entropy_tree_graphviz_dir = entropy_tree_graphviz_dir
        self.entropy_tree_graphviz_view = entropy_tree_graphviz_view
        self.entropy_tree_graphviz_max_label_chars = entropy_tree_graphviz_max_label_chars
        self.entropy_tree_graphviz_local_rank_only = entropy_tree_graphviz_local_rank_only

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def __call__(self, data: DataProto, return_dict: bool = False):
        tree_metadata = data.meta_info.get("entropy_tree_metadata")
        if tree_metadata is None:
            return self._fallback_naive(data, return_dict)

        tree_lists = _deserialize_tree_lists(tree_metadata["tree_lists"])
        tree_to_prompt_idx = tree_metadata["tree_to_prompt_idx"]

        sample_tree_idxs = data.non_tensor_batch["entropy_tree_idx"]
        sample_node_idxs = data.non_tensor_batch["entropy_node_idx"]
        sample_prompt_idxs = data.non_tensor_batch["entropy_prompt_idx"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: Dict[str, list] = defaultdict(list)

        already_print: Dict[str, int] = {}
        outcome_scores: List[float] = []
        prompts_out: List[str] = []
        gts_out: List[str] = []
        responses_out: List[str] = []

        # --- Step 1: compute outcome score for every sample (leaf) --- #
        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = item.batch["responses"]
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch[self.reward_fn_key]
            extra_info = item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if isinstance(score, dict):
                raw_score = score["score"]
                for k, v in score.items():
                    reward_extra_info[k].append(v)
            else:
                raw_score = score

            outcome_scores.append(float(raw_score))
            prompts_out.append(prompt_str)
            gts_out.append(ground_truth)
            responses_out.append(response_str)

            if data_source not in already_print:
                already_print[data_source] = 0
            if already_print[data_source] < self.num_examine:
                already_print[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", raw_score)

        # --- Step 2: assign scores to TreeNode leaves --- #
        for i, sc in enumerate(outcome_scores):
            t_idx = int(sample_tree_idxs[i])
            n_idx = int(sample_node_idxs[i])
            node = tree_lists[t_idx][n_idx]
            node.score = sc # outcome reward
            node.binary_score = 1 if sc > 0 else 0

        # --- Step 3 & 4: build value tree per prompt, back-propagate, build dense path rewards --- #
        prompt_set = sorted(set(int(x) for x in sample_prompt_idxs))
        prompt_to_trees: Dict[int, List] = {p: [] for p in prompt_set}
        for t_idx, p_idx in enumerate(tree_to_prompt_idx):
            if p_idx in prompt_to_trees:
                prompt_to_trees[p_idx].append(t_idx)

        leaf_path_reward_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

        for p_idx in prompt_set:
            group_tree_idxs = prompt_to_trees[p_idx]
            group_tree_lists = [tree_lists[t] for t in group_tree_idxs]

            root, leaves = _build_value_tree(
                group_tree_lists,
                overall_norm_style=self.overall_norm_style,
                inner_repetition_penalty=self.inner_repetition_penalty,
            )
            v_root = _node_value(root) # 累积 value / 子树中的叶子数量 -> normalized value

            for leaf in leaves:
                path = _path_from_root_to_leaf(leaf)# 从 root 到 leaf 的 value tuple
                path = _fill_in_paths(path) # 把 value 写入
                path = _apply_reward_style(path, reward_style=self.reward_style, root_value=v_root)
                real_tree_idx = group_tree_idxs[leaf.source_tree_idx]
                leaf_path_reward_map[(real_tree_idx, leaf.source_node_idx)] = path

            # Optional text / Graphviz viz after back-prop (shared seg_r map)
            lr = int(os.environ.get("LOCAL_RANK", "0"))
            do_print = (
                self.print_entropy_tree
                and leaves
                and (not self.print_entropy_tree_local_rank_only or lr == 0)
            )
            gv_dir = self.entropy_tree_graphviz_dir
            do_graphviz = (
                bool(gv_dir)
                and leaves
                and (not self.entropy_tree_graphviz_local_rank_only or lr == 0)
            )
            if do_print or do_graphviz:
                from verl.utils.entropy_tree_visualize import (
                    format_entropy_value_tree,
                    print_entropy_value_tree,
                    render_entropy_value_forest_graphviz,
                )

                seg_r_by_id: Dict[int, float] = {}
                for leaf in leaves:
                    path = _path_from_root_to_leaf(leaf)
                    path = _fill_in_paths(path)
                    path = _apply_reward_style(path, reward_style=self.reward_style, root_value=v_root)
                    nodes_chain = _value_nodes_root_to_leaf(leaf)
                    if len(nodes_chain) != len(path):
                        print(
                            "[entropy_value_tree] warn: path/nodes length mismatch "
                            f"prompt_idx={p_idx} path={len(path)} nodes={len(nodes_chain)}",
                            flush=True,
                        )
                    for nd, step in zip(nodes_chain, path):
                        seg_r_by_id[id(nd)] = float(step["value"])

                if do_print:
                    lines = format_entropy_value_tree(
                        root,
                        v_root=v_root,
                        seg_r_by_id=seg_r_by_id,
                        node_value_fn=_node_value,
                        decode_fn=lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
                        max_preview_chars=self.print_entropy_tree_max_preview_chars,
                        prompt_idx=p_idx,
                        num_group_trees=len(group_tree_idxs),
                        reward_style=self.reward_style,
                    )
                    print_entropy_value_tree(lines)

                if do_graphviz:
                    written = render_entropy_value_forest_graphviz(
                        root,
                        prompt_idx=p_idx,
                        group_tree_idxs=list(group_tree_idxs),
                        seg_r_by_id=seg_r_by_id,
                        node_value_fn=_node_value,
                        decode_fn=lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
                        output_dir=os.path.expanduser(str(gv_dir)),
                        max_label_chars=self.entropy_tree_graphviz_max_label_chars,
                        view=self.entropy_tree_graphviz_view,
                    )
                    if written:
                        print(
                            f"[entropy_value_tree] graphviz wrote {len(written)} file(s) "
                            f"for prompt_idx={p_idx} under {gv_dir!s}",
                            flush=True,
                        )

        # --- Step 5: fill reward tensor with dense segment-level rewards --- #
        for i in range(len(data)):
            t_idx = int(sample_tree_idxs[i])
            n_idx = int(sample_node_idxs[i])
            key = (t_idx, n_idx)

            prompt_length = data[i].batch["prompts"].shape[-1]
            valid_response_length = int(data[i].batch["attention_mask"][prompt_length:].sum())
            if valid_response_length <= 0:
                continue

            path = leaf_path_reward_map.get(key)
            if path is None:
                continue

            cursor = 0
            for node in path:
                step_len = len(node["token_answer"])
                if step_len <= 0:
                    continue
                end = min(cursor + step_len, valid_response_length)
                if end > cursor:
                    reward_tensor[i, cursor:end] = float(node["value"])
                cursor += step_len
                if cursor >= valid_response_length:
                    break

            if cursor < valid_response_length and len(path) > 0:
                reward_tensor[i, cursor:valid_response_length] = float(path[-1]["value"])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "prompt": prompts_out,
                "ground_truth": gts_out,
                "response": responses_out,
                "outcome_reward": outcome_scores,
            }
        return reward_tensor

    # ------------------------------------------------------------------ #
    # Fallback when tree metadata is absent
    # ------------------------------------------------------------------ #

    def _fallback_naive(self, data: DataProto, return_dict: bool):
        """Behave like NaiveRewardManager when no tree metadata is available."""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: Dict[str, list] = defaultdict(list)
        already_print: Dict[str, int] = {}

        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = item.batch["responses"]
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch[self.reward_fn_key]
            extra_info = item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if isinstance(score, dict):
                reward = score["score"]
                for k, v in score.items():
                    reward_extra_info[k].append(v)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print:
                already_print[data_source] = 0
            if already_print[data_source] < self.num_examine:
                already_print[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", reward)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
