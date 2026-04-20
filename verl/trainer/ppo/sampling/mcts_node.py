from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT_UCT_SCORE = 10_000.0
_EPSILON = 1e-10


@dataclass
class MCTSNode:
    """A node in the MCTS search tree.

    state: full prefix token ids (prompt + all accumulated step tokens).
           Fed directly to generate_fn as input_ids for next-step expansion.
    step_tokens: token ids generated at this specific step only.
    step_text: decoded text of this step (expected: one <step>…</step> block).
    accumulated_text: full response from root to this node.
    """

    state: List[int]
    step_tokens: List[int] = field(default_factory=list)
    step_text: str = ""
    accumulated_text: str = ""
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    depth: int = 0
    terminal: bool = False
    visits: int = 0
    R: float = 0.0      # immediate PRM reward for this step
    value: float = 0.0  # backed-up value used in UCT
    tree_idx: int = 0   # index into the prompt batch
    node_idx: int = 0   # unique id within this tree

    # --- ORM / correctness (set after tree is built) ---
    is_correct: Optional[bool] = None   # ORM: leaf only, None until computed
    main_chain: bool = False             # True if this node is on the correct path

    # --- Subtree statistics (propagated via leaf_backpropagate / selected_backpropagate) ---
    accumulated_value: float = 0.0       # sum of leaf accumulated_values in subtree
    terminal_in_subtree: int = 0         # total terminal nodes in subtree
    correct_terminal_in_subtree: int = 0 # correct terminal nodes in subtree
    selected_terminal_in_subtree: int = 0 # selected terminals in subtree (for weighted update)

    # --- Heuristic (can be used for selection) ---
    heuristic: float = 0.0               # = R by default; available for custom strategies

    def __hash__(self):
        return hash((self.tree_idx, self.node_idx))

    def __eq__(self, other):
        if isinstance(other, MCTSNode):
            return self.tree_idx == other.tree_idx and self.node_idx == other.node_idx
        return False


# ------------------------------------------------------------------
# Tree utilities
# ------------------------------------------------------------------

def uct(node: MCTSNode, exploration_c: float = 1.0) -> float:
    """UCT score for node selection. Root returns a sentinel high value."""
    if node.parent is None:
        return ROOT_UCT_SCORE
    return (node.value + 1.0 + 1e-8) / 2.0 + exploration_c * math.sqrt(
        math.log(node.parent.visits + 1) / (node.visits + _EPSILON)
    )


def backpropagate(node: MCTSNode, gamma: float = 0.9) -> None:
    """Walk ancestors updating value (visits-weighted Q) and visit count."""
    node.visits += 1
    if node.children:
        numer, denom = 0.0, 0
        for child in node.children:
            q = (child.R - node.R) + gamma * child.value
            numer += q * child.visits
            denom += child.visits
        node.value = numer / denom if denom else 0.0
    else:
        node.value = node.R

    if node.parent is not None:
        backpropagate(node.parent, gamma)


def leaf_normalize(leaves: List[MCTSNode]) -> None:
    """Leave-one-out normalization of immediate rewards on terminal leaves.

    Also sets accumulated_value = R for each leaf (entry value before backprop).
    """
    if len(leaves) <= 1:
        for leaf in leaves:
            leaf.accumulated_value = leaf.R
        return
    vals = [leaf.R for leaf in leaves]
    total = sum(vals)
    n = len(vals)
    for i, leaf in enumerate(leaves):
        mean_others = (total - vals[i]) / (n - 1)
        leaf.R = leaf.R - mean_others
        leaf.accumulated_value = leaf.R


def leaf_backpropagate(node: MCTSNode) -> None:
    """Propagate leaf correctness information up the tree.

    - If node is terminal + correct + main_chain: increment terminal/correct/accumulated
    - If node is terminal (any): increment terminal + accumulated
    Always walks to root.
    """
    node.terminal_in_subtree += 1
    if node.main_chain:
        node.correct_terminal_in_subtree += 1
    node.accumulated_value += node.R

    parent = node.parent
    while parent is not None:
        parent.terminal_in_subtree += 1
        if node.main_chain:
            parent.correct_terminal_in_subtree += 1
        parent.accumulated_value += node.R
        node = parent
        parent = node.parent


def selected_backpropagate(node: MCTSNode) -> None:
    """Increment selected_terminal_in_subtree for node and all ancestors."""
    node.selected_terminal_in_subtree += 1
    parent = node.parent
    while parent is not None:
        parent.selected_terminal_in_subtree += 1
        parent = parent.parent


def compute_accumulated_value(node: MCTSNode) -> float:
    """Post-order traversal: set node.accumulated_value = mean of terminal children's values."""
    if not node.children:
        return node.accumulated_value

    total = 0.0
    terminal_count = 0
    for child in node.children:
        if child.terminal_in_subtree > 0:
            total += compute_accumulated_value(child)
            terminal_count += 1

    node.accumulated_value = total / terminal_count if terminal_count else 0.0
    return node.accumulated_value


def normalize_all_steps(root: MCTSNode, style: str = "step") -> None:
    """Normalize accumulated_value across all non-root nodes.

    style="step":   accumulated_value -= mean × terminal_in_subtree
    style="token":  accumulated_value -= mean × (terminal_in_subtree × len(step_tokens))
    """
    all_nodes: List[MCTSNode] = []
    queue = deque([root])
    while queue:
        cur = queue.popleft()
        all_nodes.append(cur)
        queue.extend(cur.children)

    if style == "step":
        total_val = sum(n.accumulated_value for n in all_nodes if n.terminal_in_subtree > 0)
        total_cnt = sum(n.terminal_in_subtree for n in all_nodes if n.terminal_in_subtree > 0)
    elif style == "token":
        total_val = sum(
            n.accumulated_value * len(n.step_tokens)
            for n in all_nodes if n.terminal_in_subtree > 0
        )
        total_cnt = sum(
            n.terminal_in_subtree * len(n.step_tokens)
            for n in all_nodes if n.terminal_in_subtree > 0
        )
    else:
        return

    mean = total_val / total_cnt if total_cnt else 0.0
    for node in all_nodes:
        node.accumulated_value -= mean * node.terminal_in_subtree


def compute_weighted_update(node: MCTSNode, style: str = "uniform") -> None:
    """Divide accumulated_value by selected_terminal_in_subtree (recursively)."""
    if node.selected_terminal_in_subtree > 0:
        if style == "uniform":
            node.accumulated_value /= node.selected_terminal_in_subtree
        elif style == "sqrt":
            node.accumulated_value /= math.sqrt(node.selected_terminal_in_subtree)
        elif style == "original":
            pass
        else:
            raise ValueError(f"Unknown weighted update style: {style!r}")

    for child in node.children:
        compute_weighted_update(child, style)


def gather_path(leaf: MCTSNode) -> List[MCTSNode]:
    """Return the path from root's first child to leaf (inclusive).

    The root node (depth=0, no step_tokens) is excluded.
    Result is ordered [root_child, …, leaf].
    """
    path: List[MCTSNode] = []
    node: Optional[MCTSNode] = leaf
    while node is not None and node.parent is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    return path


def is_fully_expanded(
    node: MCTSNode,
    max_children: int,
    depth_node_count: Dict[int, int],
    max_node_per_depth: int = 16,
) -> bool:
    next_depth = node.depth + 1
    if depth_node_count.get(next_depth, 0) >= max_node_per_depth:
        return True
    return len(node.children) >= max_children or node.terminal


def collect_expandable(
    root: MCTSNode,
    depth_node_count: Dict[int, int],
    max_children: int,
    max_node_per_depth: int = 16,
) -> List[MCTSNode]:
    """BFS collection of all nodes that can still be expanded."""
    candidates: List[MCTSNode] = []
    queue = deque([root])
    while queue:
        cur = queue.popleft()
        if not is_fully_expanded(cur, max_children, depth_node_count, max_node_per_depth):
            candidates.append(cur)
        queue.extend(cur.children)
    return candidates


def select_terminal(leaves: List[MCTSNode], num_traces: int) -> List[MCTSNode]:
    """Select up to num_traces terminal leaves.

    Prioritises one main_chain leaf, then fills with random diverse parents.
    """
    if not leaves:
        return []
    if len(leaves) <= num_traces:
        return list(leaves)

    main_chain_leaf = next((l for l in leaves if l.main_chain), None)

    other_leaves = [l for l in leaves if l is not main_chain_leaf]
    random.shuffle(other_leaves)

    # Group by parent for diversity
    parent_buckets: Dict[int, List[MCTSNode]] = {}
    for leaf in other_leaves:
        key = id(leaf.parent)
        parent_buckets.setdefault(key, []).append(leaf)

    selected: List[MCTSNode] = []
    if main_chain_leaf is not None:
        selected.append(main_chain_leaf)

    parents = list(parent_buckets.keys())
    random.shuffle(parents)
    for p in parents:
        selected.append(random.choice(parent_buckets[p]))
        if len(selected) >= num_traces:
            return selected

    remaining = [l for l in other_leaves if l not in selected]
    while len(selected) < num_traces and remaining:
        chosen = random.choice(remaining)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def collect_all_nodes(root: MCTSNode) -> List[MCTSNode]:
    """BFS traversal returning all nodes in the tree."""
    result: List[MCTSNode] = []
    queue = deque([root])
    while queue:
        cur = queue.popleft()
        result.append(cur)
        queue.extend(cur.children)
    return result
