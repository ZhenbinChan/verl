# FOL / Tree-GAE Follow-Up Notes

This note summarizes the current debugging state for the LogiQA2K 1.5B FOL reward runs, so another person can continue without reconstructing the full chat history.

## Current Findings

- FOL step-GDPO v3 is the most plausible "multi-step and reasonably correct" run so far.
  - Best observed val: step 350, acc `0.38249`, val num_steps `7.11`.
  - It keeps multi-step generations, unlike tree runs.
  - Recent FOL judge metrics improved versus older v2/v2.1 runs, but leakage and sort/declaration failures still exist.
- FOL Tree-GAE shallow v3 achieved the highest current FOL-tree val acc but via short reasoning.
  - Best observed val: step 450, acc `0.43318`.
  - By the end it collapsed to very short paths, roughly one reasoning step / terminal answer.
  - Step 500 dropped to acc `0.37942`.
- FOL Tree-GAE deeper `(M,N,L,T)=(4,1,3,1)` did not solve the "multi-step and correct" target.
  - Step 50 val: acc `0.33487`, val num_steps `3.00`.
  - Step 100 val: acc `0.37174`, val num_steps `1.14`.
  - It became short as training progressed and was much slower due to FOL judge load.
- Short-tree collapse is not FOL-specific.
  - Old outcome tree quickly collapsed to `num_steps/mean ~= 2`.
  - Old self-eval tree also collapsed to `num_steps/mean ~= 2`.
  - Old format tree resisted longer, but still shortened to about `2.4-2.7` steps late in training.
- Small actor capacity is likely a real factor.
  - 1.5B learns reward shortcuts and direct-answer strategies.
  - Stable "long chain + XML-valid + FOL-verifiable + correct" behavior may require a stronger actor.

## Current Runs To Watch

- `train_fol_step_gdpo_gpu2_v3.log`
  - Step-GDPO FOL v3 on GPU2 using judge01.
  - Watch step 350/400/450/500 val acc and val num_steps.
- `train_fol_tree_gae_gpu3_judge56_v3.log`
  - FOL Tree-GAE shallow v3 on GPU3 using judge56.
  - Completed at step 500. Best observed step was 450.
- `train_fol_tree_gae_gpu4_judge56_deeper_v1.log`
  - FOL Tree-GAE deeper on GPU4 using judge56.
  - Step 100 val is already available. Continue only if there is a reason to study late collapse; otherwise it is low priority.

## Engineering Changes Already Made / Pending Commit

- `verl/utils/tree_structure.py`
  - Path-level TreeManager penalties for truncated / repeated / multi-boxed / bad-format branch paths are now consumed only when `prm_name == "fol"`.
  - This prevents FOL-specific fail-closed penalties from contaminating `format` or `self_eval` external PRM runs.
- `bash_scripts/fol_tree_gae_localjudge_boost_deeper.sh`
  - Adds a deeper FOL Tree-GAE run script.
  - Uses `(M,N,L,T)=(4,1,3,1)`:
    - `actor_rollout_ref.rollout.n=4`
    - `trainer.tree_top_n=1`
    - `trainer.tree_rounds=3`
    - `trainer.tree_branches=1`
  - Theoretical leaves per prompt: `4 * (1 + 1 * 3 * 1) = 16`, matching step-GDPO `n=16`.

## Experiment Directions

### Prompt-v2 Follow-Up

- `verl/prompts/logical_reasoning.txt` was revised to keep FOL/Z3-friendly atomic reasoning while replacing unsafe boxed examples such as `\boxed{animal}` and `\boxed{No}` with option-letter-safe examples.
- `verl/prompts/logical_reasoning_2.txt` adds a shorter prompt variant closer to the earlier premise-selection style.
- Existing `data/logiqa2k/*.parquet` files embed the old system prompt, so prompt changes require regenerating parquet data before rerunning baselines.
- For ReClor-style datasets, check both sides before training:
  - raw labels are often `0..3`;
  - the prompt should display options as A/B/C/D;
  - `reward_model.ground_truth` and reward extraction must agree on either option letters or raw indices. Prefer option letters for MCQ rewards.

### Baseline Matrix To Run

The useful experiment set is now broader than only FOL. Run it in stages so the full matrix does not waste GPU time before the prompt effect is isolated.

Stage 0: regenerate prompt-v2 data.

- Create a new data directory, e.g. `data/logiqa2k_prompt_v2/`, using `verl/prompts/logical_reasoning.txt`.
- Optionally create `data/logiqa2k_prompt_v2_short/` using `logical_reasoning_2.txt`.
- Do not overwrite the old parquet unless the old prompt is no longer needed for comparison.

Stage 1: new-prompt 1.5B sanity baselines.

- Actor: Qwen2.5-1.5B.
- Methods: step-GDPO first; tree-GAE only after step-GDPO shows nonzero/healthy behavior.
- Rewards to run:
  - outcome-only;
  - format;
  - self-eval;
  - FOL;
  - random, as a sanity/control run.
- Goal: determine whether the prompt alone fixes answer-format collapse or changes the short-reasoning tendency.

Stage 2: stronger-actor baselines.

- Actor families to compare:
  - Qwen2.5-1.5B and Qwen2.5-7B;
  - 3.5/4B and 9B class models if available locally.
- Rewards:
  - outcome-only, format, self-eval, FOL;
  - random only once per actor family as a control, not necessarily every size.
- Primary metrics:
  - val acc;
  - val num_steps;
  - XML bad-format / missing-boxed / multi-boxed rates;
  - FOL invalid translation, leakage, entailed rate, and judge latency.

Stage 3: tree reruns after step baselines.

- Only rerun Tree-GAE when the corresponding step-GDPO run is healthy.
- For each promising actor/reward pair, compare shallow tree vs deeper tree.
- Stop early if the tree collapses to one-step answer paths again without improving acc.

- Let FOL step-GDPO v3 finish first.
  - It is the main candidate for "more steps and decent accuracy".
  - Compare best checkpoint by both acc and val num_steps, not acc alone.
- Do not prioritize running FOL Tree-GAE deeper to completion.
  - It is slow and has already shortened by step 100.
  - If kept running, record whether later acc improves despite short paths.
- Run `format tree deeper` as a cheap diagnostic if GPU is available.
  - Goal: test whether deeper Tree-GAE itself can preserve tree depth when the external PRM is cheap and deterministic.
  - If format deeper also shortens, the problem is mostly Tree-GAE/reward structure rather than FOL judge.
- Consider self-eval tree only after the format deeper diagnostic.
  - Self-eval is more expensive and old self-eval tree also shortens.
  - The old self-eval tree script used penalty config that could previously contaminate non-FOL PRMs; after the FOL-only penalty fix, reruns are cleaner.
- Try a stronger actor.
  - 3B/7B is the most likely route if the goal is "multi-step + correct".
  - 1.5B appears to optimize short answer strategies more readily than long verifiable reasoning.

## Reward / Objective Ideas

- Separate two targets explicitly:
  - Outcome accuracy.
  - Process quality: enough useful steps, XML validity, and FOL-verifiable steps.
- Do not reward length blindly.
  - Format step50 had many steps but poor acc, so "more steps" alone is not useful.
- Consider a mild process-shape objective only when correctness is nonzero.
  - Example: only reward extra valid steps for samples with correct final answer, or add a small bonus for 2-5 valid reasoning steps.
  - Avoid forcing very long traces; many LogiQA examples may not need them.
- Consider a tree anti-collapse rule.
  - Penalize branches that immediately emit only boxed answer after one shallow step.
  - Keep this separate from FOL judge failure penalties.
- Consider changing Tree-GAE weights.
  - Current FOL tree uses `[0.8, 0.2]`.
  - Higher FOL/process weight may preserve process, but only after FOL success/leakage rates are good enough.

## FOL Judge / Pipeline Ideas

- Continue reducing invalid FOL translation.
  - v3 improved over v2/v2.1 substantially.
  - Remaining failure modes include unknown identifiers, sort mismatch, declaration failures, leakage, and Z3 runtime errors.
- Be careful with cumulative verification.
  - It gives stronger prefix checking but makes deeper paths more expensive and more failure-prone.
  - It also increases context length pressure for step-GDPO.
- Keep old whole-code correction off unless deliberately ablated.
  - New expression-level repair is safer than whole-program correction.
- Keep path-level penalties FOL-only.
  - `format` should be `1/0` via `check_step_format_fol`.
  - `self_eval` should use its own judge score, not FOL fail-closed penalties.

## Current Best Reference Points

| Run | Log | Status | Best Val Acc | Best Step | Val Num Steps At Best | Final / Latest Val Acc | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| FOL step-GDPO v3 | `train_fol_step_gdpo_gpu2_v3.log` | running | `0.38249` | 350 | `7.11` | `0.38249` latest known | Best current multi-step FOL run. |
| FOL Tree-GAE shallow v3 | `train_fol_tree_gae_gpu3_judge56_v3.log` | done | `0.43318` | 450 | `1.00` | `0.37942` at step 500 | Highest FOL-tree acc but short-answer strategy. |
| FOL Tree-GAE deeper v1 | `train_fol_tree_gae_gpu4_judge56_deeper_v1.log` | running | `0.37174` | 100 | `1.14` | `0.33641` at step 150 | Deeper tree still collapses short. |
| FOL step-GDPO old | `train_fol_step_gdpo_gpu2.log` | done | `0.39017` | 350 | `5.45` | `0.35791` at step 500 | Old aux FOL reward metric scale is buggy; compare acc/steps only. |
| FOL step-GDPO v2 | `train_fol_step_gdpo_gpu2_v2.log` | failed/stopped | `0.34869` | 100 | `3.97` | `0.34869` at step 100 | Hit fatal errors later. |
| FOL step-GDPO v2.1 | `train_fol_step_gdpo_gpu2_v2_1.log` | stopped | `0.34562` | 150 | `2.45` | `0.34562` at step 150 | Shorter than v3. |
| format Tree-GAE old | `train_format_tree_gae_gpu3.log` | done | `0.43318` | 450 | `2.53` | `0.40707` at step 500 | Cheap format reward; longer than FOL tree but still shortens. |
| outcome Tree-GAE old | `train_outcome_tree_gae_gpu3.log` | done | `0.35638` | 500 | `0.00` | `0.35638` at step 500 | Outcome-only tree collapse / no step metric. |
| self-eval step-GDPO old | `train_self_eval_step_gdpo_gpu3.log` | done | `0.41475` | 350 | `1.00` | `0.40553` at step 500 | Strong acc, short answers. |
| self-eval Tree-GAE old | `train_self_eval_tree_gae_gpu5_new.log` | done | `0.36098` | 150 | not recorded in latest table | `0.25960` at step 500 | Poor late behavior; old penalty contamination may affect interpretation. |

Auxiliary reward means from older runs can exceed `[0, 1]` due to an older aggregation/reporting bug. Use val acc, val num_steps, format validity, and FOL debug rates for cross-run comparison.

- Step-GDPO self-eval 8:2:
  - Log: `train_self_eval_step_gdpo_gpu3.log`
  - Best val acc: `0.41475` at step 350.
  - But val num_steps collapsed to `1.0`.
- FOL Tree-GAE shallow v3:
  - Log: `train_fol_tree_gae_gpu3_judge56_v3.log`
  - Best val acc: `0.43318` at step 450.
  - But it is a short-answer strategy.
- FOL step-GDPO v3:
  - Log: `train_fol_step_gdpo_gpu2_v3.log`
  - Best observed val acc so far: `0.38249` at step 350.
  - Keeps val num_steps around `7`.

## Useful Local Commands

Summarize current FOL runs:

```bash
python - <<'PY'
import re
from pathlib import Path
files = [
    ("step", "train_fol_step_gdpo_gpu2_v3.log"),
    ("tree_shallow", "train_fol_tree_gae_gpu3_judge56_v3.log"),
    ("tree_deeper", "train_fol_tree_gae_gpu4_judge56_deeper_v1.log"),
]
keys = [
    "training/global_step",
    "critic/score/mean",
    "step_gdpo/fol_step_reward/mean",
    "fol_judge/invalid_translation_rate/mean",
    "fol_judge/leakage_rate/mean",
    "fol_judge/entailed_rate/mean",
    "tree/avg_leaves_per_tree",
    "tree/pass_rate",
    "num_steps/mean",
    "response_length/mean",
    "timing_s/step",
]
step_re = re.compile(r"step:(\d+) -")
def get(line, key):
    m = re.search(re.escape(key) + r":([-+0-9.eE]+)", line)
    return None if not m else float(m.group(1))
for name, file in files:
    p = Path(file)
    print("\n", name, file)
    if not p.exists():
        print("missing")
        continue
    step_lines = []
    val_lines = []
    for line in p.open(errors="ignore"):
        if step_re.search(line):
            step_lines.append(line)
        if "val-core/logiqa/acc/mean@1" in line:
            val_lines.append(line)
    if step_lines:
        line = step_lines[-1]
        print("latest", step_re.search(line).group(1))
        for key in keys:
            value = get(line, key)
            if value is not None:
                print(key, value)
    for line in val_lines[-5:]:
        print("val", step_re.search(line).group(1) if step_re.search(line) else "?", get(line, "val-core/logiqa/acc/mean@1"), get(line, "val-aux/logiqa/num_steps/mean@1"))
PY
```
