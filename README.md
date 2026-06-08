# verl for Tree Search RL

This repository is a verl fork focused on RL training for reasoning tasks, with extra support for tree-style rollout expansion, step-level process reward, FOL/Z3 verification, and logic QA datasets such as LogiQA and ReClor.

## What Is Added

- Pluggable rollout expansion strategies under `trainer.sampling_strategy`: plain rollout, legacy tree search, entropy-chain TreeRL, parallel MCTS, Step-TreeRL, and information-gain expansion.
- Step-level process reward through `trainer.process_reward.type=format|fol`, shared by Step-TreeRL, parallel MCTS, and information-gain sampling.
- FOL-as-PRM utilities for translating/verifying logic reasoning steps with an OpenAI-compatible, MiniMax, or Azure OpenAI LLM backend plus Z3.
- Dataset preprocessors and launch scripts for GSM8K, LogiQA, ReClor, MCQ-style data, FOL metadata, GRPO, and Step-TreeRL experiments.
- Extra training metrics for tree rollouts, especially Step-TreeRL trace counts, leaf accuracy, format ratio, and timing.

## Repository Map

- `verl/trainer/config/ppo_trainer.yaml`: main PPO/GRPO configuration entry.
- `verl/trainer/ppo/sampling/`: rollout expansion strategy implementations.
- `verl/workers/reward_manager/`: reward managers for plain, tree, entropy, MCTS, Step-TreeRL, and information-gain workflows.
- `verl/utils/process_reward.py`: canonical process-reward config and runtime builder.
- `examples/data_preprocess/`: dataset and FOL metadata preprocessing scripts.
- `bash_scripts/logiqa/`, `bash_scripts/reclor/`, `bash_scripts/TreeSearch/`: runnable experiment scripts.
- `CONFIG.md`: detailed configuration guide.

## Installation

### Prerequisites

- Python >= 3.11
- CUDA >= 12.4
- PyTorch 2.6.0 recommended
- Ray 2.48.0
- vLLM 0.8.5.post1 recommended

### Setup

```bash
conda create -n verl_plus python=3.11
conda activate verl_plus

git clone https://github.com/BiNLP/verl
cd verl
pip install -e .
pip install -r requirements.txt
```

Recommended CUDA 12.4 stack:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5.post1
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

## Dataset Preparation

### GSM8K

```bash
python3 examples/data_preprocess/gsm8k.py --local_dir data/gsm8k
```

### LogiQA / ReClor

```bash
python3 examples/data_preprocess/logiqa.py --local_dir data/logiqa
python3 examples/data_preprocess/reclor.py --local_dir data/reclor
```

### MCQ Preprocessor

Use `mcq_preprocess.py` when converting existing multiple-choice parquet files or preparing FOL metadata.

```bash
python examples/data_preprocess/mcq_preprocess.py \
  --input_parquet data/reclor/train.parquet \
  --output_dir data/reclor \
  --preset reclor \
  --skip_fol_extraction
```

For FOL-as-PRM metadata:

```bash
python examples/data_preprocess/mcq_preprocess.py \
  --input_parquet data/reclor/train.parquet \
  --output_dir data/reclor_fol \
  --preset reclor \
  --base_url "http://localhost:4869/v1" \
  --model "qwen2.5-7b-coder" \
  --max_retries -1 \
  --verbose
```

Expected outputs:

- `train.parquet` and `test.parquet`: training/validation data.
- `fol_metadata.json`: required when `trainer.process_reward.type=fol` and offline metadata is used.

### Global FOL PRM Metadata

Use the split metadata preprocessor when preparing FOL PRM data for Step-TreeRL. It converts dataset splits separately and writes both split-specific metadata and one merged metadata file.

```bash
bash bash_scripts/preprocess/global_fol_prm_metadata_splits.sh \
  --api_config llm_server/configs/deepseek.yaml
```

The script defaults to ReClor:

- input: `data/reclor`
- output: `data/reclor_global_fol_prm`
- splits: discovered from `train.parquet`, `test.parquet`, and validation aliases if present
- metadata: `fol_metadata_train.json`, `fol_metadata_test.json`, and merged `fol_metadata_all.json`

For a small API smoke test:

```bash
bash bash_scripts/preprocess/global_fol_prm_metadata_splits.sh \
  --api_config llm_server/configs/deepseek.yaml \
  --output_dir /tmp/verl_fol_deepseek_reclor_smoke \
  --splits train,test \
  --num_samples_per_split 1 \
  --max_workers 1 \
  --max_retries 3 \
  --save_every 1
```

Provider settings are loaded from YAML or JSON files under `llm_server/configs/`. For example, `llm_server/configs/deepseek.yaml` contains the OpenAI-compatible endpoint, model name, request defaults, optional `extra_body`, and the `api_key` field used by both preprocessing and online FOL verification.

## Training Modes

The default entrypoint is:

```bash
python3 -m verl.trainer.main_ppo key=value ...
```

### Plain GRPO

Use this for normal response-level reward training without tree expansion.

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.rollout.n=4 \
  reward_model.reward_manager=auto \
  trainer.sampling_strategy=null
```

Example scripts:

- `bash_scripts/logiqa/Qwen3-8B-base_GRPO_base.sh`
- `bash_scripts/reclor/Qwen3-8B-base_GRPO_base.sh`
- `examples/grpo_trainer/run_qwen2-7b.sh`

### Step-TreeRL

Use this for step-level tree expansion: generate complete initial rollouts, split them by `<step>...</step>`, select high-entropy step nodes, branch from selected nodes, backpropagate leaf correctness/value, then train on selected terminal traces.

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=step_treerl_grpo \
  actor_rollout_ref.actor.policy_loss=tree_loss \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.rollout.n=6 \
  reward_model.reward_manager=auto \
  trainer.sampling_strategy=step_treerl \
  trainer.process_reward.type=format \
  trainer.step_treerl_config.m=6 \
  trainer.step_treerl_config.n=2 \
  trainer.step_treerl_config.l=1 \
  trainer.step_treerl_config.t=2 \
  trainer.step_treerl_config.selected_num_traces=16
```

Example scripts:

- `bash_scripts/logiqa/Qwen3-8B-base_StepTreeRL_format_reward.sh`
- `bash_scripts/logiqa/Qwen3-8B-base_StepTreeRL_fol_reward.sh`
- `bash_scripts/reclor/Qwen3-8B-base_StepTreeRL_format_reward.sh`
- `bash_scripts/reclor/Qwen3-8B-base_StepTreeRL_fol_reward.sh`

### Other Tree Strategies

| Strategy | `trainer.sampling_strategy` | Reward manager | Advantage estimator | Main config |
| --- | --- | --- | --- | --- |
| Legacy tree search | `tree_search` | `tree` | `tree_grpo` or `tree_gae` | `trainer.tree_rounds`, `tree_top_k`, `branch_level` |
| Entropy-chain TreeRL | `treerl` | `entropy` | `entropy_reinforce` | `trainer.entropy_chain_config` |
| Parallel MCTS | `parallel_mcts` | `mcts` | `mcts_grpo` | `trainer.parallel_mcts_config` |
| Step-TreeRL | `step_treerl` | `step_tree` | `step_treerl_grpo` or `step_treerl_reinforce` | `trainer.step_treerl_config` |
| Information gain | `information_gain` | `ig` | `ig_grpo` | `trainer.ig_config` |

`reward_model.reward_manager=auto` resolves the manager from `trainer.sampling_strategy` for the tree strategies above.

## Process Reward

Process reward lives under `trainer.process_reward`, not `reward_model.reward_kwargs`.

### Format PRM

```bash
trainer.process_reward.type=format
```

This checks whether reasoning steps follow the expected step format. It is the simplest process reward and is usually enough for smoke tests.

## Rollout Format Metrics

Rollout format metrics are trainer-level diagnostics for prompts that ask the model to emit explicit `<step>...</step>` reasoning and a final boxed multiple-choice answer. Enable them with:

```bash
trainer.log_format_metrics=True
```

The LogiQA GRPO scripts also expose this as:

```bash
LOG_FORMAT_METRICS=True bash bash_scripts/logiqa/Qwen2.5-7B_LogiQA_GRPO_only.sh
```

These metrics are computed after rollout expansion is finished and after the final training trajectories have been assembled. This means they apply to plain GRPO rollouts and to expanded traces from `tree_search`, `treerl`, `parallel_mcts`, `step_treerl`, and `information_gain`. They are independent of the reward manager. Keep `trainer.log_format_metrics=False` for prompts such as `prompts/base.txt`, where the step/premise/conclusion format is not expected.

For validation with `naive_plus`, the same switch also records the `step_tree`-style auxiliary fields `format_full`, `format_answer_only`, `format_step_only`, and `format_trace_total`, which appear as `val-aux/{data_source}/format_*/...` metrics.

### Training Metrics

| Field | Meaning |
| --- | --- |
| `rollout/format_primary/total` | Number of rollout trajectories included in the format statistics for this training step. |
| `rollout/format_primary/full_ratio` | Fraction of trajectories whose step blocks and final boxed answer are both valid. |
| `rollout/format_primary/relax_correct_ratio` | Fraction whose step XML/schema and final boxed answer are valid when arbitrary text outside complete step blocks is ignored. |
| `rollout/format_primary/no_step_ratio` | Fraction of trajectories with no `<step>...</step>` block in the reasoning region. |
| `rollout/format_primary/text_outside_step_ratio` | Fraction of trajectories that contain complete step blocks, but also contain non-whitespace text outside those step blocks before the final answer. |
| `rollout/format_primary/step_xml_invalid_ratio` | Fraction of trajectories with malformed step XML, such as an opened `<step>` tag without a valid closing block. |
| `rollout/format_primary/step_schema_invalid_ratio` | Fraction of trajectories whose step XML parses, but the step schema is invalid. |
| `rollout/format_primary/boxed_missing_ratio` | Fraction of trajectories with valid step blocks but no final `\boxed{...}` answer. |
| `rollout/format_primary/boxed_invalid_ratio` | Fraction of trajectories with valid step blocks and a `\boxed` answer region, but the answer format is invalid. |
| `rollout/answer_acc/all_correct_ratio` | Answer accuracy over all rollout trajectories whose answer correctness can be read from the reward manager. |
| `rollout/answer_acc/format_correct_only_ratio` | Answer accuracy after removing format-incorrect trajectories; this is computed only over trajectories with `format_primary=full`. |

The primary category ratios are mutually exclusive and sum to 1 for a non-empty rollout batch. `relax_correct_ratio` is a derived metric and is not part of that sum.

`reward/mean_fn_reward` is the mean final training reward, while `rollout/answer_acc/...` records answer correctness. For `naive_plus`, format-incorrect trajectories receive `-1` reward at the last valid response token only when `reward_model.reward_kwargs.penalize_format_error=True`, so `reward/mean_fn_reward` can differ from answer accuracy. The default is `False` to preserve `prompts/base.txt` training behavior.

```bash
reward_model.reward_kwargs.penalize_format_error=True
```

### Format Rules

A trajectory is counted as `full` only when all of the following hold:

- The reasoning region contains one or more complete `<step>...</step>` blocks.
- There is no text outside the step blocks before the final answer other than real whitespace or the literal whitespace escapes `\n`, `\r`, `\t`, `\v`, and `\f`.
- Each step is valid XML with root tag `step`.
- Each step contains at least one `<premise>...</premise>` and exactly one `<conclusion>...</conclusion>`.
- Step children are only `premise` or `conclusion`; nested tags, direct text under `<step>`, and non-whitespace child tails are invalid.
- The final answer is a strict trailing boxed answer with exactly one alphabetic index, for example `\boxed{A}` or `\boxed{{A}}`.

Examples of invalid boxed answers include `\boxed{A}}`, `\boxed{{A}`, `\boxed{AA}`, `\boxed{AB}`, `\boxed{1}`, and an empty box.

For `relax_correct_ratio`, arbitrary text outside complete step blocks is ignored, but at least one complete step, valid step XML/schema, no unmatched step tags, and a valid trailing boxed answer are still required. Literal whitespace escapes after the boxed answer are not ignored because the boxed answer must remain the strict end of the response.

### Rollout JSONL Fields

When `trainer.rollout_data_dir` is set and `trainer.log_format_metrics=True`, dumped rollout JSONL rows include these per-trajectory fields:

| Field | Meaning |
| --- | --- |
| `format_primary` | The mutually exclusive category assigned to this trajectory: `full`, `no_step`, `text_outside_step`, `step_xml_invalid`, `step_schema_invalid`, `boxed_missing`, or `boxed_invalid`. |
| `boxed_status` | `valid`, `invalid`, or `missing`, describing only the final boxed answer region. |
| `boxed_answer` | The extracted answer letter when `boxed_status=valid`; otherwise an empty string. |
| `step_block_count` | Number of complete `<step>...</step>` blocks found before the final answer region. |
| `answer_acc` | Per-trajectory answer correctness, when provided by the reward manager. |

### FOL PRM

```bash
trainer.process_reward.type=fol \
trainer.process_reward.fol.prm_mode=global_fol_prm \
trainer.process_reward.fol.metadata_path=/path/to/fol_metadata.json \
trainer.process_reward.fol.llm.api_config=llm_server/configs/deepseek.yaml
```

FOL scoring needs `sample_id` metadata in the batch. For global FOL PRM, use the merged split metadata file such as `data/reclor_global_fol_prm/fol_metadata_all.json`, so train, test, and validation IDs are resolved consistently. If metadata is missing and `online_declaration_fallback=true`, the runtime can generate declarations online through the configured LLM backend.

The FOL LLM config can still be overridden from the command line, but the recommended path is to keep provider-specific settings in `llm_server/configs/*.yaml` and pass only `trainer.process_reward.fol.llm.api_config=...` in training scripts.

### ReClor Step-TreeRL FOL Script

The main ReClor FOL Step-TreeRL script is:

```bash
bash_scripts/reclor/Qwen3-8B-base_StepTreeRL_fol_reward.sh
```

It uses `FOL_API_CONFIG` to select the provider config. Example:

```bash
FOL_API_CONFIG=/home/chenzhb/Workspaces/verl/llm_server/configs/deepseek.yaml \
bash bash_scripts/reclor/Qwen3-8B-base_StepTreeRL_fol_reward.sh
```

The recent 1-step smoke test used the same script with overrides:

```bash
FOL_API_CONFIG=/home/chenzhb/Workspaces/verl/llm_server/configs/deepseek.yaml \
bash bash_scripts/reclor/Qwen3-8B-base_StepTreeRL_fol_reward.sh \
  actor_rollout_ref.model.path=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct \
  data.train_files=/tmp/verl_fol_deepseek_reclor_direct_key_train2/train.parquet \
  data.val_files=/tmp/verl_fol_deepseek_reclor_direct_key_train2/test.parquet \
  data.train_batch_size=2 \
  data.max_prompt_length=1024 \
  data.max_response_length=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_model_len=1536 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.step_treerl_config.max_depth=1 \
  trainer.step_treerl_config.max_token_num=256 \
  trainer.step_treerl_config.branch_max_new_tokens=64 \
  trainer.step_treerl_config.m=1 \
  trainer.step_treerl_config.n=1 \
  trainer.step_treerl_config.l=0 \
  trainer.step_treerl_config.t=1 \
  trainer.step_treerl_config.selected_num_traces=1 \
  trainer.process_reward.fol.metadata_path=/tmp/verl_fol_deepseek_reclor_direct_key_train2/fol_metadata_all.json \
  trainer.process_reward.fol.max_retries=1 \
  trainer.process_reward.fol.verify_timeout=10 \
  trainer.process_reward.fol.llm.max_concurrency=1 \
  trainer.logger="['console']" \
  trainer.experiment_name=StepTreeRL_Reclor_FOL_deepseek_direct_key_smoke \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=1 \
  trainer.total_epochs=1
```

In that run, the training step completed and logged `reward/step_treerl_process_reward_mean`. The final checkpoint save failed because the workspace filesystem was full, not because of FOL metadata or provider config loading.

## Evaluation

Use the bundled evaluation scripts:

```bash
sh bash_scripts/eval/eval_lighteval.sh
sh bash_scripts/eval/eval_QA_lighteval.sh
```

Dataset-specific helpers:

- `bash_scripts/eval/Qwen2.5-1.5B_LogiQA_eval.sh`
- `bash_scripts/eval/Qwen2.5-1.5B_ReClor_eval.sh`
- `bash_scripts/eval/Qwen2.5-7B_LogiQA_eval.sh`
- `bash_scripts/eval/Qwen2.5-7B_ReClor_eval.sh`

### Cross-Domain MCQ Evaluation

Cross-domain evaluation uses local parquet files plus the repository's two-stage evaluation flow:

1. `verl.trainer.main_generation` generates model responses into `eval_output/main_eval/.../*_generated.parquet`.
2. `verl.trainer.main_eval` scores the generated responses with `bash_scripts/eval/custom_module.py`, which delegates to `verl.utils.reward_score.default_compute_score`.

The current Qwen3-8B-Base scripts use:

```bash
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen3-8B-Base
```

Prepare or refresh the cross-domain parquet files:

```bash
python3 examples/data_preprocess/pubmedqa.py \
  --data_dir ./data/pubmedqa_origin/data \
  --local_dir ./data/pubmedqa/

python3 examples/data_preprocess/truthfulqa.py \
  --local_dir ./data/truthfulqa/

python3 examples/data_preprocess/qa4mre.py \
  --local_dir ./data/qa4mre/

python3 examples/data_preprocess/gpqa.py \
  --local_dir ./data/gpqa/

python3 examples/data_preprocess/mathqa.py \
  --data_dir ./data/MathQA \
  --local_dir ./data/mathqa/

python3 examples/data_preprocess/openbookqa.py \
  --local_dir ./data/openbookqa/

python3 examples/data_preprocess/medqa.py \
  --local_dir ./data/medqa/
```

Expected evaluation files:

| Dataset | Parquet |
| --- | --- |
| PubMedQA | `data/pubmedqa/test.parquet` |
| TruthfulQA MC1 | `data/truthfulqa/test.parquet` |
| QA4MRE 2013 EN | `data/qa4mre/test.parquet` |
| GPQA Diamond | `data/gpqa/gpqa_diamond/test.parquet` |
| GPQA Main | `data/gpqa/gpqa_main/test.parquet` |
| MathQA | `data/mathqa/test.parquet` |
| MathQA Challenge | `data/mathqa/challenge_test.parquet` |
| OpenBookQA | `data/openbookqa/test.parquet` |
| MedQA | `data/medqa/test.parquet` |

Run one dataset-specific Qwen3-8B-Base evaluation script:

```bash
bash bash_scripts/eval/qwen3-8b-base_pubmedqa_eval.sh
bash bash_scripts/eval/qwen3-8b-base_truthfulqa_eval.sh
bash bash_scripts/eval/qwen3-8b-base_qa4mre_eval.sh
bash bash_scripts/eval/qwen3-8b-base_gpqa_diamond_eval.sh
bash bash_scripts/eval/qwen3-8b-base_gpqa_main_eval.sh
bash bash_scripts/eval/qwen3-8b-base_mathqa_eval.sh
bash bash_scripts/eval/qwen3-8b-base_mathqa_challenge_eval.sh
bash bash_scripts/eval/qwen3-8b-base_openbookqa_eval.sh
bash bash_scripts/eval/qwen3-8b-base_medqa_eval.sh
```

The shared runner can also be called directly with a dataset name:

```bash
bash bash_scripts/eval/qwen3-8b-base_eval_common.sh pubmedqa
bash bash_scripts/eval/qwen3-8b-base_eval_common.sh mathqa_challenge
```

Running the shared runner without an argument only prints the available dataset names. Outputs are written under:

```bash
eval_output/main_eval/qwen3_8b_base_<dataset>/
```

Each output directory contains:

- `<dataset>_generated.parquet`: generated responses.
- `<dataset>_main_eval.log`: final reward/accuracy log printed by `main_eval`.

For a quick smoke test, edit the target script or common runner and set:

```bash
MAX_SAMPLES=2
```

Then run the dataset script. Full evaluation uses `MAX_SAMPLES=0`.

## Practical Notes

- For Step-TreeRL, prefer prompts that force explicit `<step>...</step>` segmentation; otherwise branch extraction and format PRM become noisy.
- For GRPO and Step-TreeRL, `actor_rollout_ref.rollout.n` controls initial samples per prompt. Step-TreeRL also has `m`; keep `m` aligned with `rollout.n` unless you intentionally override it.
- For long or variable traces, prefer dynamic batch: `actor_rollout_ref.actor.use_dynamic_bsz=True` and tune `ppo_max_token_len_per_gpu`.
- For Step-TreeRL, set `actor_rollout_ref.rollout.max_model_len >= data.max_prompt_length + trainer.step_treerl_config.max_token_num`.
- Do not commit real API keys. Use environment interpolation such as `${oc.env:MINIMAX_API_KEY}` in launch scripts.
