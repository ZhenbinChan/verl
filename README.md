# verl: Volcano Engine Reinforcement Learning for LLMs (Forked)

This is a customized fork of [verl](https://github.com/volcengine/verl) tailored for logical reasoning tasks, process reward-based reinforcement learning methods (Step-GDPO, parallel generation or tree search), and specialized dataset preprocessing pipelines & prompting for LogiQA datasets.

For the original verl library's detailed documentation and features, please refer to [README-bytedance.md](README-bytedance.md).

---

## LogiQA Dataset Preprocessing & Prompting

The LogiQA dataset preprocessing allows injecting custom reasoning instructions (e.g., `p1 & p2 -> i1` for step-wise logical inferences) via flexible prompt file configurations.

### Version, Format, # of samples, and Output Directory

You can customize the LogiQA dataset loading and preprocessing by configuring a few parameters in `logiqa.py`:

- `--version`: Specifies LogiQA dataset version (`1` for `lucasmccabe/logiqa` or `2` for `baber/logiqa2`). Default is `1`.
- `--num_samples`: The number of training samples to keep. Use `-1` for all samples. Default is `2000`.
- `--local_save_dir`: The directory to save the output `.parquet` files. Default is `./data/logiqa2k`.
- `--format`: Prompt formatting style. Default is `flat`.
  - `flat`: Regular plain text format (`Context: ...\n\nQuestion: ...\n\nOptions: ...`).
  - `xml`: XML tag format (`<Context>...\n</Context>\n<Question>...`).

**Example:**

Version 1, 2000 samples, plain text format:

```bash
python examples/data_preprocess/logiqa.py \
    --version 1 \
    --num_samples 2000 \
    --local_save_dir ./data/logiqa2k \
    --system_prompt_file logical_reasoning.txt
```

Version 1, all samples, plain text format:

```bash
python examples/data_preprocess/logiqa.py \
    --version 1 \
    --num_samples -1 \
    --local_save_dir ./data/logiqa \
    --system_prompt_file logical_reasoning.txt
```

Version 2, 5000 samples, XML format:

```bash
python examples/data_preprocess/logiqa.py \
    --version 2 \
    --num_samples 5000 \
    --format xml \
    --local_save_dir ./data/logiqa5k_v2_xml \
    --system_prompt_file logical_reasoning.txt
```

Injection of Logic Reasoning Prompt:

```bash
python examples/data_preprocess/logiqa.py \
    --version 2 \
    --num_samples 5000 \
    --format xml \
    --local_save_dir ./data/logiqa5k_v2_xml \
    --system_prompt_file logical_reasoning.txt
```

### Prompting

```bash
# 只加 system prompt（读取 verl/prompts/logical_reasoning.txt）
python examples/data_preprocess/logiqa.py \
    --system_prompt_file logical_reasoning.txt

# 只加 user prompt（在题目后追加）
python examples/data_preprocess/logiqa.py \
    --user_prompt_file logical_reasoning.txt

# 两个都加（system + user 各用不同的 txt）
python examples/data_preprocess/logiqa.py \
    --system_prompt_file my_system.txt \
    --user_prompt_file my_user_instructions.txt

# 传绝对路径也支持
python examples/data_preprocess/logiqa.py \
    --system_prompt_file /path/to/any_prompt.txt
```

## Training Parameters

### Step / (External) Process Reward

### Tree Search

## Training Scripts

### DAPO

DAPO is the original baseline method explored prior to Step-GDPO. It mitigates mode collapse via an overlong-buffer mechanism.

#### Sanity Check

```bash
bash bash_scripts/sanity_check_dapo.sh
```

#### One Epoch Training

```bash
bash bash_scripts/one_epoch_dapo.sh
```

### Step-GDPO + Parallel Sampling

Step-GDPO is the core algorithm currently under development, leveraging First-Order Logic (FOL) API evaluations as step-wise rewards during training.

#### Sanity Check with Random Reward

Useful for validating the local training loop with a dummy random reward provider:

```bash
bash bash_scripts/sanity_check_step_gdpo.sh
```

#### One Epoch Training with FOL Reward

Set up the OpenAI-compatible API details for remote FOL step evaluation:

```bash
export OPENAI_API_KEY="sk-YOUR-KEY-HERE"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export FOL_MODEL="gpt-4o-mini-2024-07-18"

bash bash_scripts/fol_step_gdpo.sh
```

### Step-GDPO + TreeRL (Entropy-guided Branching Tree Search) Sampling

TODO: Tree search configurations and documentation to be added.

## Slurm Integration

The repository is built to work flexibly with Slurm workloads. You can use `srun` to submit your jobs. Here is an example of running the GDPO sanity check on a single A800 GPU:

```bash
srun -p gpu_a800 -G1 bash -c "export PYTHONUNBUFFERED=1; bash bash_scripts/sanity_check_step_gdpo.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
```

# Scripts of Baseline / Methods Walkthrough

> 所有脚本位于 `bash_scripts/`，统一使用 **Qwen2.5-1.5B-Instruct** 模型、**logiqa2k** 数据集、1 GPU 单节点、1 epoch 训练。

---

## 1. 脚本总览

| 类别 | 脚本 | adv_estimator | reward_manager | step_reward_type | rollout.n | 外部依赖 |
|------|------|---------------|----------------|------------------|-----------|----------|
| **DAPO** | `one_epoch_dapo.sh` | grpo | dapo | — | 16 | 无 |
| | `sanity_check_dapo.sh` | grpo | dapo | — | 16 | 无 |
| **Step-GDPO** | `fol_step_gdpo.sh` | step_gdpo | step | fol | 16 | OpenAI API |
| | `format_step_gdpo.sh` | step_gdpo | step | format | 16 | 无 |
| | `sanity_check_step_gdpo.sh` | step_gdpo | step | fol | 16 | OpenAI API |
| **Tree-GAE** | `format_tree_gae.sh` | tree_gae | tree | format | 6 | 无 |
| | `outcome_tree_gae.sh` | tree_gae | tree | —(纯 outcome) | 6 | 无 |
| | `sanity_check_tree_gae.sh` | tree_gae | tree | — | 6 | 无 |
| **Self-Eval** | `self_eval_step_gdpo_local.sh` | step_gdpo | step | self_eval | 16 | 本地 vLLM (GPU 1) |
| | `self_eval_step_gdpo_remote.sh` | step_gdpo | step | self_eval | 16 | 远程 API |
| | `self_eval_tree_gae_local.sh` | tree_gae | tree | self_eval | 6 | 本地 vLLM (GPU 1) |
| | `self_eval_tree_gae_remote.sh` | tree_gae | tree | self_eval | 6 | 远程 API |

---

## 2. DAPO Baseline（对照组）

DAPO 是最基础的 GRPO baseline，用于对照实验。

### 核心配置

```bash
algorithm.adv_estimator=grpo
reward_model.reward_manager=dapo
```

### 特有参数：overlong_buffer

DAPO 必须开启超长惩罚，否则模型会出现模式崩溃（重复生成 token）：

```bash
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
+reward_model.reward_kwargs.overlong_buffer_cfg.len=512        # 缓冲区长度
+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
+reward_model.reward_kwargs.max_resp_len=2048
```

惩罚逻辑：如果响应长度超过 `max_resp_len - overlong_buffer_len`（即 2048 - 512 = 1536 token），则按超出比例扣分。

### 脚本

| 脚本 | 用途 | 训练步数 | WandB |
|------|------|----------|-------|
| `one_epoch_dapo.sh` | 完整 1 epoch 训练 | 全量 | 开启 |
| `sanity_check_dapo.sh` | 快速验证 | 5 步 | 关闭 |

---

## 3. Step-GDPO

Step-GDPO 在 GRPO 基础上引入**步级奖励**，将每个推理步骤独立评分，而非只看最终答案。

### 与 DAPO 的关键差异

```diff
- algorithm.adv_estimator=grpo
+ algorithm.adv_estimator=step_gdpo

- reward_model.reward_manager=dapo
+ reward_model.reward_manager=step

+ algorithm.step_reward_type=fol|format       # 步级奖励类型
+ algorithm.step_reward_weights=[0.5, 0.5]    # [outcome_weight, process_weight]
+ algorithm.use_xml_steps=true                 # 用 XML 标签解析步骤边界

- overlong_buffer_cfg (DAPO 特有，Step-GDPO 不需要)
```

### 双 reward 权重机制

`step_reward_weights=[0.5, 0.5]` 控制 outcome reward 和 process reward 的混合比例：

- 第一个权重 (0.5)：结果正确性（最终答案是否正确）
- 第二个权重 (0.5)：过程质量（每步推理的 fol/format 分数）

### 脚本

| 脚本 | step_reward_type | 说明 |
|------|------------------|------|
| `fol_step_gdpo.sh` | fol | FOL 一阶逻辑奖励，需要 OpenAI API (gpt-4o-mini) |
| `format_step_gdpo.sh` | format | 纯格式奖励，无外部依赖 |
| `sanity_check_step_gdpo.sh` | fol | 快速验证（5 步，含 FOL API 配置） |

### FOL API 环境变量（仅 fol 模式需要）

```bash
export OPENAI_API_KEY=${OPENAI_API_KEY:-"sk-YOUR-KEY-HERE"}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-"https://api.openai.com/v1"}
export FOL_MODEL=${FOL_MODEL:-"gpt-4o-mini-2024-07-18"}
```

---

## 4. Tree-GAE（TreeRL 树搜索）

Tree-GAE 基于 EPTree（arXiv:2506.11902）实现树搜索 RL 训练。与 Step-GDPO 的"线性推理链"不同，Tree-GAE 在推理过程中进行分叉搜索，通过树结构探索更多推理路径。

### 与 Step-GDPO 的关键差异

```diff
- algorithm.adv_estimator=step_gdpo
+ algorithm.adv_estimator=tree_gae

- reward_model.reward_manager=step
+ reward_model.reward_manager=tree

- rollout.n=16
+ rollout.n=6    # 树会分叉扩展，实际评估路径数远大于 6

+ trainer.tree_sampling=True
+ trainer.tree_rounds=1          # 树搜索轮数 L
+ trainer.tree_top_n=2           # 每轮选 top-N 节点扩展
+ trainer.tree_branches=2        # 每节点分叉数 T
+ trainer.tree_mask_tail_ratio=0.1
```

### EPTree 参数

当前配置为 **(M=6, N=2, L=1, T=2)**：

- **M=6**：初始采样 6 条响应（`rollout.n=6`）
- **N=2**：每轮选 top-2 节点 (`tree_top_n=2`)
- **L=1**：1 轮树搜索 (`tree_rounds=1`)
- **T=2**：每节点 2 个分支 (`tree_branches=2`)
- 最终产生约 **30 条叶子路径** 用于 advantage 计算

### Advantage Pipeline 参数

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `tree_step_reward_mode` | la | ga_la / ga / value_only | 步级奖励模式（la = local advantage） |
| `tree_overall_norm_style` | token | step / none | 归一化粒度 |
| `tree_use_weighted_value` | False | True | 是否使用加权 value |
| `tree_weighted_value_style` | sqrt | uniform / original | 加权方式（仅 use_weighted_value=True 时生效） |
| `tree_ext_reward_dedup` | True | False | 去重共享前缀的 ext PRM 分数 |

### step_reward_weights 含义变化

在 Tree-GAE 中，`step_reward_weights=[0.5, 0.5]` 的语义变为：

- 第一个权重 (0.5)：树结构内生 advantage（基于 GA+LA 的叶子正确率推导）
- 第二个权重 (0.5)：外部 PRM 奖励（format / fol / self_eval）

### 脚本

| 脚本 | step_reward_type | 说明 |
|------|------------------|------|
| `format_tree_gae.sh` | format | 树搜索 + format 外部 PRM |
| `outcome_tree_gae.sh` | 无 | 纯 outcome，退化为 (GA+LA)/sqrt(n) 作为唯一 advantage |
| `sanity_check_tree_gae.sh` | — | 快速验证（5 步） |

---

## 5. Self-Eval 步级奖励（新增）

Self-Eval 是新增的步级奖励模式，使用 LLM（通常是参考模型本身）对每个推理步骤进行 0-10 评分，归一化到 [0, 1]。

### 核心实现

**文件**：`verl/utils/reward_score/self_eval.py`

```
compute_step_reward_self_eval(step_text, prompt_text, step_history, ...)
    -> 判断是否为终止步（包含 \boxed{}）
    -> 选择对应的 system prompt (terminal / non_terminal)
    -> 将累积推理历史拼接为 user prompt
    -> 调用 LLM API 评分
    -> 正则提取 "Overall Score: <float>"
    -> 返回 score / 10.0，范围 [0, 1]
```

### 评分 Rubric

**非终止步** (`verl/prompts/self_eval/non_terminal.txt`)：评估中间推理步骤

| 维度 | 分值 | 说明 |
|------|------|------|
| Premise Establishment | 0-2 | 前提信息和假设的清晰度 |
| Step Validity | 0-2 | 每步逻辑是否有效、格式良好 |
| Justification Quality | 0-2 | 是否引用了规则/公理/推理依据 |
| Logical Progression | 0-2 | 步骤间过渡是否流畅，无跳跃 |
| Conclusion | 0-2 | 当前步结论是否从前提中正确推出 |

满分 10 分，各维度均匀分布。

**终止步** (`verl/prompts/self_eval/terminal.txt`)：评估包含最终答案的步骤

| 维度 | 分值 | 说明 |
|------|------|------|
| Premise Establishment | 0-1 | 前提信息（降权） |
| Step Validity | 0-2 | 步骤有效性 |
| Justification Quality | 0-1 | 推理依据（降权） |
| Logical Progression | 0-2 | 逻辑连贯性 |
| **Conclusion** | **0-4** | **最终结论是否正确解决问题（加权）** |

满分 10 分，结论维度加权到 4 分（占 40%），因为终止步的核心价值是最终答案的正确性。

### 两种部署模式

#### Mode A：远程 API（1 GPU）

训练和 self-eval 共用同一 GPU，评分请求发送到远程 API：

```bash
export OPENAI_BASE_URL="https://your-remote-server/v1"
export OPENAI_API_KEY="your-key"
export SELF_EVAL_MODEL="Qwen2.5-1.5B-Instruct"   # 可选
bash self_eval_step_gdpo_remote.sh
```

#### Mode B：本地 vLLM（2 GPU）

GPU 0 跑训练，GPU 1 启动 vLLM 服务充当评分 API：

```bash
export CUDA_VISIBLE_DEVICES=0,1
bash self_eval_step_gdpo_local.sh
```

local 脚本会自动：

1. 在 GPU 1 启动 vLLM server（端口默认 8199）
2. 等待 server 就绪（最多 60 秒）
3. 设置 `OPENAI_BASE_URL=http://localhost:8199/v1`
4. 在 GPU 0 启动训练
5. 训练结束后自动 kill vLLM 进程（trap EXIT）

### API 环境变量优先级

```
reward.api_config（CLI 参数）> 环境变量 > 默认值
```

| 环境变量 | 回退 | 默认值 | 说明 |
|----------|------|--------|------|
| `SELF_EVAL_MODEL` | `FOL_MODEL` | `gpt-4o-mini` | 评分模型名称 |
| `OPENAI_API_KEY` | — | `""` | API 密钥（本地用 `EMPTY`） |
| `OPENAI_BASE_URL` | — | `None` | API 端点 |

### 脚本

| 脚本 | 训练框架 | 部署模式 | GPU 需求 |
|------|----------|----------|----------|
| `self_eval_step_gdpo_local.sh` | Step-GDPO | 本地 vLLM | 2 GPU |
| `self_eval_step_gdpo_remote.sh` | Step-GDPO | 远程 API | 1 GPU |
| `self_eval_tree_gae_local.sh` | Tree-GAE | 本地 vLLM | 2 GPU |
| `self_eval_tree_gae_remote.sh` | Tree-GAE | 远程 API | 1 GPU |

### Reward Manager 集成

Self-eval 在 `StepRewardManager` 和 `TreeRewardManager` 中均通过懒加载注册：

```python
# verl/experimental/reward_loop/reward_manager/step.py:122-125
# verl/experimental/reward_loop/reward_manager/tree.py:136-139
if "self_eval" in self.step_reward_types:
    from verl.utils.reward_score.self_eval import compute_step_reward_self_eval
    if "self_eval" not in self.step_reward_fns:
        self.step_reward_fns["self_eval"] = compute_step_reward_self_eval
```

与 fol / format 的注册方式完全一致，即插即用。

---

## 6. 三种步级奖励类型对比

| 维度 | format | fol | self_eval |
|------|--------|-----|-----------|
| **计算方式** | 正则匹配 XML 格式 | Z3 求解器验证逻辑可满足性 | LLM 按 rubric 评分 |
| **返回值** | 二值 0.0 / 1.0 | 连续 [0, 1] | 连续 [0, 1]（10分制/10） |
| **外部依赖** | 无 | OpenAI API + Z3 | OpenAI 兼容 API |
| **延迟** | 极低（纯文本匹配） | 中（API 调用） | 中（API 调用） |
| **终止步检测** | 不区分 | 不区分 | 区分（\boxed{} 启发式，结论加权） |
| **步骤历史** | 只看当前步 | 只看问题上下文 | 传入完整累积推理历史 |
| **适用场景** | 验证输出格式规范 | 逻辑推理题 (LogiQA) | 通用推理任务 |
| **可用训练框架** | Step-GDPO / Tree-GAE | Step-GDPO | Step-GDPO / Tree-GAE |

---

## 7. 公共超参数

以下参数在所有脚本中保持一致：

| 参数 | 值 | 说明 |
|------|-----|------|
| `model.path` | Qwen2.5-1.5B-Instruct | 基础模型 |
| `data` | logiqa2k (train + validation) | 数据集 |
| `max_prompt_length` | 2048 | 最大 prompt 长度 |
| `max_response_length` | 2048 | 最大响应长度 |
| `actor.optim.lr` | 1e-6 | 学习率 |
| `actor.use_kl_loss` | True | 开启 KL 散度损失 |
| `actor.kl_loss_coef` | 0.02 | KL 系数 |
| `actor.kl_loss_type` | low_var_kl | 低方差 KL |
| `rollout.temperature` | 0.8 | 采样温度 |
| `rollout.top_p` | 0.95 | top-p 采样 |
| `rollout.gpu_memory_utilization` | 0.5 | vLLM 显存占比 |
| `use_kl_in_reward` | False | reward 中不加 KL |
| `total_epochs` | 1 | 总训练轮次 |
| `test_freq` | 100 | 测试频率（步） |
| `n_gpus_per_node` | 1 | 每节点 GPU 数 |
