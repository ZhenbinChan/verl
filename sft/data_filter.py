#!/usr/bin/env python3
"""data_filter.py

从 all_wrong_uids 中挑选 200 个 unique ID，
从 train.parquet 中提取这些题目保存为 JSONL，
其余数据保存为新的 train.parquet。
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # verl 仓库根目录
WRONG_UIDS_FILE = ROOT / "rollout" / "qwen3-8b_steprl_base" / "qwen3-8b_steprl_base.json"
TRAIN_PARQUET = ROOT / "data" / "logiqa" / "train.parquet"
VAL_PARQUET = ROOT / "data" / "logiqa" / "validate.parquet"

SFT_DIR = ROOT / "sft"
SELECTED_JSONL = SFT_DIR / "selected_200.jsonl"
REMAINING_PARQUET = SFT_DIR / "train_remaining.parquet"

NUM_SELECT = 200
SEED = 42


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def load_wrong_uids() -> list[str]:
    """加载并去重 all_wrong_uids。"""
    with open(WRONG_UIDS_FILE) as f:
        data = json.load(f)
    raw = data.get("all_wrong_uids", [])
    # 去重并保持相对顺序
    unique = list(dict.fromkeys(raw))
    print(f"[data_filter] all_wrong_uids: {len(raw)} entries, {len(unique)} unique")
    return unique


def load_val_ids() -> set[str]:
    """加载 validate 集中的 sample_id，用于排除。"""
    if not VAL_PARQUET.exists():
        print("[data_filter] validate.parquet not found, skip val-exclusion")
        return set()
    table = pq.read_table(str(VAL_PARQUET))
    df = table.to_pandas()
    ids = set(df["sample_id"].tolist())
    print(f"[data_filter] validate set: {len(ids)} unique IDs")
    return ids


def main():
    SFT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载 wrong uids
    wrong_uids = load_wrong_uids()

    # 2. 加载 validate ids 用于排除
    val_ids = load_val_ids()

    # 3. 从 wrong_uids 中排除已在 validate 中的
    candidate_pool = [uid for uid in wrong_uids if uid not in val_ids]
    print(f"[data_filter] candidate pool (wrong & not-in-val): {len(candidate_pool)}")

    if len(candidate_pool) < NUM_SELECT:
        print(f"[data_filter] WARNING: only {len(candidate_pool)} candidates, using all of them")
        selected = candidate_pool
    else:
        random.seed(SEED)
        selected = random.sample(candidate_pool, NUM_SELECT)
    selected_set = set(selected)
    print(f"[data_filter] selected {len(selected)} IDs")

    # 4. 读取 train.parquet
    train_table = pq.read_table(str(TRAIN_PARQUET))
    train_df = train_table.to_pandas()
    print(f"[data_filter] train.parquet: {len(train_df)} rows")

    # 5. 筛选
    mask_selected = train_df["sample_id"].isin(selected_set)
    df_selected = train_df[mask_selected].copy()
    df_remaining = train_df[~mask_selected].copy()

    print(f"[data_filter] selected rows: {len(df_selected)}")
    print(f"[data_filter] remaining rows: {len(df_remaining)}")

    if len(df_selected) < NUM_SELECT:
        print(f"[data_filter] WARNING: only matched {len(df_selected)} rows for {NUM_SELECT} IDs")

    # 6. 保存 selected 为 JSONL（处理 numpy 类型）
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return super().default(o)

    records = df_selected.to_dict(orient="records")
    with open(SELECTED_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, cls=NpEncoder) + "\n")
    print(f"[data_filter] saved {len(records)} records to {SELECTED_JSONL}")

    # 7. 保存 remaining 为 parquet
    remaining_table = pa.Table.from_pandas(df_remaining, schema=train_table.schema)
    pq.write_table(remaining_table, str(REMAINING_PARQUET))
    print(f"[data_filter] saved {len(df_remaining)} rows to {REMAINING_PARQUET}")


if __name__ == "__main__":
    main()
