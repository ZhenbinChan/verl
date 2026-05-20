# 配置笔记

主配置入口是 `verl/trainer/config/ppo_trainer.yaml`。训练脚本一般通过 Hydra 风格的 `key=value` 覆盖默认值。

这份笔记主要记录 GRPO、Tree/Step-TreeRL、process reward、batch/token、loss 和长度预算相关配置。

## 总体流程

一次训练 step 先从 `data.train_batch_size` 个 prompt 开始，然后 rollout 对每个 prompt 采样 `actor_rollout_ref.rollout.n` 条初始 response。

```text
初始 response 数 = data.train_batch_size * actor_rollout_ref.rollout.n
```

如果 `trainer.sampling_strategy` 不为空，初始 response 会继续被树型策略扩展，最终进入 PPO/GRPO 更新的样本数取决于具体策略：

```text
普通 GRPO:        train_batch_size * rollout.n
tree_search:      train_batch_size * rollout.n * ((tree_rounds + 1) * tree_top_k)
parallel_mcts:    train_batch_size * rollout.n * parallel_mcts_config.num_traces
step_treerl:      约等于 train_batch_size * selected_num_traces，另有 GPU padding
information_gain: 策略输出的 leaves，另有 GPU padding
```

树型策略测试时先把 `data.train_batch_size` 设小。prompt 数很小也可能扩展出很多 terminal traces。

## 策略组合

| 场景 | `trainer.sampling_strategy` | 需要的 `reward_model.reward_manager` | 需要的 `algorithm.adv_estimator` | process reward |
| --- | --- | --- | --- | --- |
| 普通 PPO/GRPO | `null` | `auto` 或普通 manager | `gae`、`grpo` 等 | 可选 |
| 旧版 tree search | `tree_search` | `tree` | `tree_grpo` 或 `tree_gae` | `trainer.step_reward_type` |
| Entropy-chain TreeRL | `treerl` | `entropy` | `entropy_reinforce` | 通常不用 |
| Parallel MCTS | `parallel_mcts` | `mcts` | `mcts_grpo` | `format` 或 `fol` |
| Step-TreeRL | `step_treerl` | `step_tree` | `step_treerl_grpo` 或 `step_treerl_reinforce` | `format` 或 `fol` |
| Information Gain | `information_gain` | `ig` | `ig_grpo` | `format` 或 `fol` |

可以直接设：

```bash
reward_model.reward_manager=auto
```

代码会按策略自动映射：

```text
tree_search -> tree
treerl -> entropy
step_treerl -> step_tree
parallel_mcts -> mcts
information_gain -> ig
默认 -> naive
```

## data 配置

```yaml
data:
  train_files: ~/data/rlhf/gsm8k/train.parquet
  val_files: ~/data/rlhf/gsm8k/test.parquet
  prompt_key: prompt
  reward_fn_key: data_source
  max_prompt_length: 512
  max_response_length: 512
  train_batch_size: 1024
  filter_overlong_prompts: false
  truncation: error
  prompt_path: null
```

关键点：

- `max_prompt_length` 限制 prompt token 长度。
- `max_response_length` 控制普通 rollout 最多生成多少 token，默认传给 `actor_rollout_ref.rollout.response_length`。
- `prompt_path` 会在训练时给 prompt 追加 instruction。
- FOL process reward 需要稳定的 `sample_id`，预处理脚本通常会放在 `sample_id` 或 `extra_info` 里。

## actor / rollout / ref

### model

```yaml
actor_rollout_ref:
  model:
    path: ~/models/deepseek-llm-7b-chat
    enable_gradient_checkpointing: true
    use_remove_padding: false
    use_liger: false
    trust_remote_code: false
```

长 trace 或长短差异大的 batch，通常建议开：

```bash
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.model.use_remove_padding=True
```

### actor

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: null
    use_dynamic_bsz: false
    ppo_max_token_len_per_gpu: 16384
    policy_loss: null
    loss_agg_mode: token-mean
    use_kl_loss: false
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
    entropy_coeff: 0
    ppo_epochs: 1
```

推荐：

```bash
# 普通 GRPO，并且希望 loss_agg_mode 真正作用到 policy loss
actor_rollout_ref.actor.policy_loss=vanilla
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean

# Step-TreeRL
actor_rollout_ref.actor.policy_loss=tree_loss
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
```

注意：如果 `policy_loss=null`，会走默认 PPO loss 路径。这个路径的 policy loss 主项不完全使用 `loss_agg_mode`。如果你在意 loss 聚合方式，建议显式设 `policy_loss=vanilla` 或 `policy_loss=tree_loss`。

### rollout

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    mode: sync
    temperature: 1.0
    top_k: -1
    top_p: 1.0
    response_length: ${data.max_response_length}
    n: 1
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.5
    max_num_batched_tokens: 8192
    max_model_len: null
```

关键点：

- `rollout.n` 是每个 prompt 初始采样几条 response。
- GRPO 通常需要 `rollout.n > 1`。
- Step-TreeRL 里通常让 `rollout.n` 和 `trainer.step_treerl_config.m` 保持一致。
- `max_model_len` 是推理引擎上下文上限，也就是 prompt + generated tokens。

### ref

`actor_rollout_ref.ref.*` 只控制 reference logprob 前向，不训练。一般跟 actor 的 dynamic batch 设置保持一致。如果 ref logprob OOM，就单独降低 `ref.log_prob_max_token_len_per_gpu`。

## critic

PPO/GAE 需要 critic。GRPO 和 Step-TreeRL 通常不依赖 critic value learning，但配置仍然存在。

```yaml
critic:
  ppo_mini_batch_size: ${actor_rollout_ref.actor.ppo_mini_batch_size}
  ppo_micro_batch_size_per_gpu: null
  use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
  ppo_max_token_len_per_gpu: 32768
  forward_max_token_len_per_gpu: ${critic.ppo_max_token_len_per_gpu}
```

如果 critic OOM，优先降：

```bash
critic.ppo_max_token_len_per_gpu=16384
critic.forward_max_token_len_per_gpu=16384
```

## batch 和 token 调参

PPO 更新分两层：

```text
ppo_mini_batch_size              PPO optimizer 的全局 mini-batch
ppo_micro_batch_size_per_gpu     固定模式下每张 GPU 每次 forward/backward 的 sequence 数
ppo_max_token_len_per_gpu        dynamic batch 模式下每张 GPU 每个 micro-batch 的 token 上限
```

固定 micro batch：

```bash
actor_rollout_ref.actor.use_dynamic_bsz=False
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
```

dynamic token batch：

```bash
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
```

Tree/Step-TreeRL 更建议 dynamic batch，因为 trace 长短差异通常很大。

估算：

```text
ppo_max_token_len_per_gpu ~= 每卡目标 sequence 数 * 平均 sequence 长度
sequence 长度 ~= max_prompt_length + max_response_length
```

例子：

```text
512 prompt + 512 response，每卡约 8 条   -> 8192
512 prompt + 512 response，每卡约 16 条  -> 16384
2048 prompt + 4096 response，树型 trace -> 先从 8192 或 12000 试
```

OOM 处理顺序：

1. 降 `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`。
2. 降 rollout/ref logprob 的 token budget。
3. 降 `data.train_batch_size`、`rollout.n` 或树型策略的 trace 数。
4. 开 gradient checkpointing / offload。
5. 如果是 generation 阶段卡住，再降 `max_model_len`、`max_response_length` 或树扩展预算。

## loss_agg_mode

允许值：

```bash
actor_rollout_ref.actor.loss_agg_mode=token-mean
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
```

含义：

- `token-mean`：所有有效 token 一起平均。长 response 权重大。
- `seq-mean-token-sum`：每条 response 内部 token loss 求和，再对 response 平均。长 response 梯度可能更大。
- `seq-mean-token-mean`：每条 response 内部 token loss 平均，再对 response 平均。通常最适合 GRPO 和 Step-TreeRL。
- `seq-mean-token-sum-norm`：每条 response token loss 求和，再除以固定 response length 维度，接近 Dr.GRPO/DAPO 风格。

推荐：

```bash
# 普通 GRPO
actor_rollout_ref.actor.policy_loss=vanilla
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean

# Step-TreeRL
actor_rollout_ref.actor.policy_loss=tree_loss
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean

# Dr.GRPO / DAPO 风格
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
algorithm.norm_adv_by_std_in_grpo=False
```

## 长度预算

这两个参数控制不同阶段：

```bash
data.max_response_length=4096
actor_rollout_ref.rollout.max_model_len=8192
```

`data.max_response_length` 控制最多生成多少新 token。`rollout.max_model_len` 控制推理引擎上下文窗口：

```text
prompt tokens + generated tokens <= max_model_len
```

普通 rollout 的安全关系：

```text
actor_rollout_ref.rollout.max_model_len >= data.max_prompt_length + data.max_response_length
```

Step-TreeRL 的安全关系：

```text
actor_rollout_ref.rollout.max_model_len >= data.max_prompt_length + trainer.step_treerl_config.max_token_num
```

Step-TreeRL 分支生成时还会看当前 node 已经有多长：

```text
remaining_context_budget = max_model_len - len(current_node_state)
branch_budget = min(branch_max_new_tokens, remaining_response_budget, remaining_context_budget)
```

如果当前路径已经接近 `max_model_len`，后续分支会被缩短或跳过。

## Step-TreeRL 配置

配置在 `trainer.step_treerl_config`：

```yaml
trainer:
  sampling_strategy: step_treerl
  process_reward:
    type: format
  step_treerl_config:
    max_depth: 40
    max_token_num: 512
    branch_max_new_tokens: 512
    m: 6
    n: 2
    l: 1
    t: 2
    path_selection: selected_terminals
    selected_num_traces: 16
    use_weighted_value: true
    weighted_value_style: sqrt
    overall_norm_style: none
    length_penalty:
      enabled: true
      p_max: 0.1
      k: 15.0
      t0: 0.7
```

参数含义：

- `m`：每个 prompt 初始完整 rollout 数。通常等于 `actor_rollout_ref.rollout.n`。
- `n`：每棵初始 rollout tree 每轮选择多少个高熵 branch point。
- `l`：branching rounds。
- `t`：每个被选 branch point 继续采样多少条 continuation。
- `max_token_num`：一条完整轨迹的 response token 预算。
- `branch_max_new_tokens`：单次 branch continuation 的硬上限，之后还会被剩余预算 clamp。
- `path_selection`：`selected_terminals` 或 `all_leaves`。
- `selected_num_traces`：每个 prompt 最后选多少条 terminal trace 进训练。
- `use_weighted_value`：backprop 时是否使用 weighted value。
- `weighted_value_style`：`sqrt`、`uniform` 或 `original`。
- `overall_norm_style`：`none`、`step` 或 `token`。

推荐 smoke 配置：

```bash
algorithm.adv_estimator=step_treerl_grpo
actor_rollout_ref.actor.policy_loss=tree_loss
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
actor_rollout_ref.rollout.n=6
reward_model.reward_manager=auto
trainer.sampling_strategy=step_treerl
trainer.process_reward.type=format
trainer.step_treerl_config.m=6
trainer.step_treerl_config.n=2
trainer.step_treerl_config.l=1
trainer.step_treerl_config.t=2
trainer.step_treerl_config.selected_num_traces=16
```

FOL reward 配置：

```bash
trainer.process_reward.type=fol
trainer.process_reward.fol.prm_mode=global_fol_prm
trainer.process_reward.fol.metadata_path=/path/to/fol_metadata.json
trainer.process_reward.fol.online_declaration_fallback=true
trainer.process_reward.fol.fail_on_missing_metadata=false
trainer.process_reward.fol.llm.provider=openai_compatible
trainer.process_reward.fol.llm.api_base_url=http://localhost:4869/v1
trainer.process_reward.fol.llm.api_key=EMPTY
trainer.process_reward.fol.llm.model_name=qwen2.5-7b-coder
trainer.process_reward.fol.llm.max_concurrency=8
```

## Parallel MCTS 配置

```yaml
trainer:
  sampling_strategy: parallel_mcts
  process_reward:
    type: format
  parallel_mcts_config:
    max_nodes: 20
    max_depth: 40
    max_children: 3
    concurrent_num: 4
    pass_k: 4
    num_traces: 4
    exploration_constant: 1.0
    gamma: 0.9
    max_token_num: 512
    backprop: true
    random_pick: false
    selection_policy: importance_sampling
    use_weighted_value: false
    normalize_style: step
    average_one_generation: false
```

配套：

```bash
trainer.sampling_strategy=parallel_mcts
algorithm.adv_estimator=mcts_grpo
reward_model.reward_manager=auto
trainer.process_reward.type=format
```

## Information Gain 配置

Information Gain 是 KL-guided step branching 策略。

```yaml
trainer:
  sampling_strategy: information_gain
  process_reward:
    type: format
  ig_config:
    max_depth: 40
    max_token_num: 512
    prefix_text: "So far, the most likely answer is \\boxed{"
    top_k: 2
    iter_rounds: 3
```

配套：

```bash
trainer.sampling_strategy=information_gain
algorithm.adv_estimator=ig_grpo
reward_model.reward_manager=auto
trainer.process_reward.type=format
```

## Entropy-chain TreeRL 配置

```yaml
trainer:
  sampling_strategy: treerl
  entropy_chain_config:
    N: 4
    L: 3
    T: 1
    max_token_num: 4096
    evaluation_strategy: token-entropy
    enforce_uniform_per_prompt: true
```

配套：

```bash
trainer.sampling_strategy=treerl
algorithm.adv_estimator=entropy_reinforce
reward_model.reward_manager=auto
```

## 旧版 tree_search 配置

```yaml
trainer:
  sampling_strategy: tree_search
  branch_level: step
  step_reward_type: random
  tree_rounds: 1
  tree_top_k: 1
```

配套：

```bash
trainer.sampling_strategy=tree_search
algorithm.adv_estimator=tree_grpo
reward_model.reward_manager=auto
```

`branch_level=token` 对长推理序列会生成非常大的树。除非明确需要 token-level node，否则优先用 `step`。

## process_reward 配置

现在 process reward 的规范位置是：

```yaml
trainer:
  process_reward:
    type: none
    fol:
      prm_mode: global_fol_prm
      metadata_path: null
      online_declaration_fallback: true
      fail_on_missing_metadata: false
      verify_timeout: 10.0
      max_retries: 3
      debug_dir: null
      llm:
        provider: openai_compatible
        api_base_url: http://localhost:4869/v1
        api_key: EMPTY
        model_name: null
        max_tokens: 4096
        temperature: 0.1
        top_p: 0.8
        max_concurrency: 8
        request_timeout: 60
```

支持的 `type`：

- `none`：不用 process reward。
- `format`：检查 step 结构和格式。
- `fol`：用 FOL/Z3 验证 step。

FOL PRM mode：

- `global_fol_prm`：用累计 reasoning text 和全局 FOL context 验证。
- `local_fol_prm`：单步验证，依赖已有或在线生成的 declaration。

FOL 失败行为：

- `online_declaration_fallback=true`：metadata 缺失时允许在线生成 declaration。
- `fail_on_missing_metadata=false`：缺失或验证失败时给 `0.0`，不中断训练。
- `fail_on_missing_metadata=true`：缺失 metadata 或验证错误时直接失败。

## 2 卡 / 4 卡切换

树型训练在 2 卡和 4 卡之间切换时，dynamic token batch 更稳：

```bash
data.train_batch_size=4
actor_rollout_ref.rollout.n=6
actor_rollout_ref.actor.ppo_mini_batch_size=4
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
```

注意这个关系：

```text
actor.ppo_mini_batch_size * rollout.n >= 总 GPU 数
```

最好让相关 batch 数能被 GPU 数整除。

## 推荐调参顺序

1. 先定长度：`data.max_prompt_length`、`data.max_response_length`、`rollout.max_model_len`、策略里的 `max_token_num`。
2. 再定初始采样量：`data.train_batch_size` 和 `actor_rollout_ref.rollout.n`。
3. 选择 rollout 策略和对应的 advantage estimator。
4. 选择 process reward：smoke test 先用 `format`，FOL metadata 和后端准备好后再用 `fol`。
5. 开 dynamic batch，找到稳定的 `ppo_max_token_len_per_gpu`。
6. 基础 run 稳定后，再加 `selected_num_traces`、`m`、`n`、`l`、`t`。
7. 最后再调 KL、学习率、loss aggregation、weighted value 等训练细节。
