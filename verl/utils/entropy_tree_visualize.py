"""Text / Graphviz visualization for entropy-chain value trees after back-prop.

Used by ``EntropyRewardManager`` only *after* ``_build_value_tree`` and value
normalization. Nodes are duck-typed (``answer_token``, ``children``, ``parent``,
``R``, ``accumulated_value``, ``terminal_in_subtree``, ``terminal``, optional
``source_tree_idx``, ``source_node_idx``, ``finish_reason``).

PNG export follows the same idea as ``mcts_utils/tree_node.visualize_tree`` but
targets ``_ValueNode`` and avoids importing ``mcts_utils`` (heavy side effects).
Needs the ``graphviz`` pip package and a system ``dot`` binary. For Weights &
Biases, log the saved files with ``wandb.Image(path)`` from your training script
if desired.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

_GRAPHVIZ_WARNED = False


def _truncate_preview(text: str, max_chars: int) -> str:
    text = text.replace("\n", "\\n")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def format_entropy_value_tree(
    virtual_root: Any,
    *,
    v_root: float,
    seg_r_by_id: Dict[int, float],
    node_value_fn: Callable[[Any], float],
    decode_fn: Callable[[List[int]], str],
    max_preview_chars: int,
    prompt_idx: int,
    num_group_trees: int,
    reward_style: str,
) -> List[str]:
    """DFS topology: one line per node with V, R, seg_r, acc, t_in, preview."""
    lines: List[str] = []
    header = (
        f"[entropy_value_tree] prompt_idx={prompt_idx} "
        f"group_trees={num_group_trees} v_root={v_root:.6g} reward_style={reward_style}"
    )
    lines.append(header)

    def fmt_segment_line(node: Any, depth: int, *, is_virtual: bool) -> str:
        indent = "  " * depth
        v = node_value_fn(node) if not is_virtual else float(v_root)
        acc = float(getattr(node, "accumulated_value", 0.0))
        t_in = int(getattr(node, "terminal_in_subtree", 0))
        r = float(getattr(node, "R", 0.0))
        term = bool(getattr(node, "terminal", False))
        st = getattr(node, "source_tree_idx", -1)
        sn = getattr(node, "source_node_idx", -1)
        fr = getattr(node, "finish_reason", None)
        toks = getattr(node, "answer_token", []) or []
        preview = _truncate_preview(decode_fn(list(toks)), max_preview_chars) if toks else ""
        seg_r = seg_r_by_id.get(id(node))
        seg_s = f"{seg_r:.6g}" if seg_r is not None else "-"
        tail = f" finish={fr}" if fr else ""
        raw = getattr(virtual_root, "reward_raw", None) if is_virtual else None
        raw_s = f" reward_raw={raw:.6g}" if is_virtual and raw is not None else ""
        prefix = "<virtual_root>" if is_virtual else "|-"
        return (
            f"{indent}{prefix} V={v:.6g} R={r:.6g} seg_r={seg_s} "
            f"acc={acc:.6g} t_in={t_in} term={term} src=t{st}:n{sn}{tail}{raw_s} | {preview}"
        )

    lines.append(fmt_segment_line(virtual_root, 0, is_virtual=True))

    def dfs(node: Any, depth: int) -> None:
        for ch in getattr(node, "children", []) or []:
            lines.append(fmt_segment_line(ch, depth + 1, is_virtual=False))
            dfs(ch, depth + 1)

    dfs(virtual_root, 0)
    return lines


def print_entropy_value_tree(lines: List[str]) -> None:
    for ln in lines:
        print(ln, flush=True)


def render_entropy_value_forest_graphviz(
    virtual_root: Any,
    *,
    prompt_idx: int,
    group_tree_idxs: List[int],
    seg_r_by_id: Dict[int, float],
    node_value_fn: Callable[[Any], float],
    decode_fn: Callable[[List[int]], str],
    output_dir: str,
    max_label_chars: int = 48,
    view: bool = False,
) -> List[str]:
    """One PNG per top-level tree under ``virtual_root``: ``{prompt_idx}_{global_tree_idx}.png``.

    ``group_tree_idxs[local_i]`` maps ``_ValueNode.source_tree_idx`` to the global
    tree index used in batch metadata (``entropy_tree_idx``).
    """
    global _GRAPHVIZ_WARNED
    try:
        from graphviz import Digraph
    except ImportError:
        if not _GRAPHVIZ_WARNED:
            print(
                "[entropy_value_tree] graphviz: pip package `graphviz` not installed; "
                "skipping PNG export.",
                flush=True,
            )
            _GRAPHVIZ_WARNED = True
        return []

    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    def node_label(n: Any) -> str:
        v = float(node_value_fn(n))
        r = float(getattr(n, "R", 0.0))
        acc = float(getattr(n, "accumulated_value", 0.0))
        t_in = int(getattr(n, "terminal_in_subtree", 0))
        term = bool(getattr(n, "terminal", False))
        st = int(getattr(n, "source_tree_idx", -1))
        sn = int(getattr(n, "source_node_idx", -1))
        fr = getattr(n, "finish_reason", None) or ""
        seg_r = seg_r_by_id.get(id(n))
        seg_s = f"{seg_r:.4g}" if seg_r is not None else "-"
        toks = getattr(n, "answer_token", []) or []
        preview = ""
        if toks:
            preview = _truncate_preview(decode_fn(list(toks)), max_label_chars).replace('"', "'")
        lines = [
            f"V={v:.4g} R={r:.4g} seg_r={seg_s}",
            f"acc={acc:.4g} t_in={t_in} term={term}",
            f"t{st}:n{sn} {fr}".strip(),
        ]
        if preview:
            lines.append(preview)
        return "\n".join(lines)

    def build_and_render(sub_root: Any, global_tree_idx: int) -> None:
        # 必须声明 global：否则 except 里对 _GRAPHVIZ_WARNED 赋值会让整段函数把它当局部变量，
        # 从而在「先读再赋」分支触发 UnboundLocalError。
        global _GRAPHVIZ_WARNED
        dot = Digraph(comment=f"entropy_p{prompt_idx}_t{global_tree_idx}", format="png")
        dot.attr(rankdir="TB")

        def walk(n: Any, parent_id: Optional[str]) -> str:
            nid = f"n_{id(n)}"
            dot.node(nid, label=node_label(n), shape="box", fontsize="9")
            if parent_id is not None:
                dot.edge(parent_id, nid)
            for ch in getattr(n, "children", []) or []:
                walk(ch, nid)
            return nid

        walk(sub_root, None)
        basename = os.path.join(output_dir, f"{prompt_idx}_{global_tree_idx}")
        try:
            out = dot.render(basename, cleanup=True, view=view)
        except Exception as e:
            if not _GRAPHVIZ_WARNED:
                print(
                    f"[entropy_value_tree] graphviz render failed ({e!r}); "
                    "is the `dot` binary installed and the `graphviz` pip package present?",
                    flush=True,
                )
                _GRAPHVIZ_WARNED = True
            return
        if out:
            written.append(out)
        else:
            written.append(basename + ".png")

    for sub_root in getattr(virtual_root, "children", []) or []:
        local_i = int(getattr(sub_root, "source_tree_idx", -1))
        if local_i < 0 or local_i >= len(group_tree_idxs):
            global_tid = local_i
        else:
            global_tid = int(group_tree_idxs[local_i])
        build_and_render(sub_root, global_tid)

    return written
