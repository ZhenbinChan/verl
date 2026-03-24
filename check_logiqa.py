import pandas as pd
import numpy as np

# 读取两个文件
df_good = pd.read_parquet("/data/home/scyb676/run/work/verl/data/logiqa/train.parquet")
df_bad = pd.read_parquet("/data/home/scyb676/run/work/verl/data/logiqa2k/train.parquet")

print("=== 数据列名对比 ===")
print("全量数据 columns:", df_good.columns.tolist())
print("2k 数据 columns:", df_bad.columns.tolist())

print("\n=== 检查是否有空值 ===")
print("2k 数据 prompt 空值数量:", df_bad['prompt'].isna().sum())
if 'data_source' in df_bad.columns:
    print("2k 数据 data_source 空值数量:", df_bad['data_source'].isna().sum())

# === 新增一个用于美化打印 Prompt 的辅助函数 ===
def print_prompt(prompt_data):
    # 处理 pandas 中可能是 numpy array 或 list 的 dict 数据
    if isinstance(prompt_data, (list, np.ndarray)):
        for msg in prompt_data:
            role = msg.get('role', 'Unknown')
            content = msg.get('content', '')
            print(f"[{role.upper()}]:\n{content}\n")
    # 兼容处理 HuggingFace dataset 转换 pandas 时可能出现的 dict of lists 格式
    elif isinstance(prompt_data, dict) and 'role' in prompt_data:
        roles = prompt_data['role']
        contents = prompt_data['content']
        for r, c in zip(roles, contents):
            print(f"[{r.upper()}]:\n{c}\n")
    else:
        print("未知的 prompt 格式:", type(prompt_data), prompt_data)

# 我们指定查看第 1233 条数据（注意 2k 数据可能没有 1233 条，所以加个判断）
idx_to_check = 1233 if len(df_bad) > 1233 else 0

print(f"\n=== 打印第 {idx_to_check} 条数据进行肉眼对比 ===")

print("\n>>>>>>>>[全量数据的 Prompt] <<<<<<<<")
print_prompt(df_good['prompt'].iloc[idx_to_check])

print("\n>>>>>>>> [2k 数据的 Prompt] <<<<<<<<")
print_prompt(df_bad['prompt'].iloc[idx_to_check])

# 注意：Reward Target 存在于 reward_model 列中，而不是 data_source
print("\n=== 检查 Reward Target (Ground Truth) ===")
print("[全量数据的 Reward Target]:", df_good['reward_model'].iloc[idx_to_check])
print("[2k 数据的 Reward Target]:", df_bad['reward_model'].iloc[idx_to_check])

# check ['extra_info'] if exists
if 'extra_info' in df_good.columns and 'extra_info' in df_bad.columns:
    print("\n=== 检查 extra_info 列 ===")
    print("全量数据 extra_info 空值数量:", df_good['extra_info'].isna().sum())
    print("2k 数据 extra_info 空值数量:", df_bad['extra_info'].isna().sum())
    
    # 打印单条数据的 extra_info，而不是把整列打出来
    print("\n[全量数据的 extra_info (单条)]:\n", df_good['extra_info'].iloc[idx_to_check])
    print("\n[2k 数据的 extra_info (单条)]:\n", df_bad['extra_info'].iloc[idx_to_check])