# Repository Guidelines

## Project Structure & Module Organization

`verl/` is the main Python package for RL/LLM training, including trainers, workers, rollout backends, models, utilities, prompts, and version metadata. `recipe/` and `examples/` contain runnable training, evaluation, and preprocessing workflows. `tests/` holds pytest suites split by area, including `ray_cpu`, `ray_gpu`, `gpu_utility`, `trainer`, `workers`, and `utils`. `docs/` contains documentation sources, while `bash_scripts/` stores experiment and evaluation shell entrypoints. `mcts_utils/` and `llm_server/` provide tree-search and API inference utilities. Treat `data/`, `outputs/`, `wandb/`, `ckpt/`, and `eval_output/` as local artifacts.

## Build, Test, and Development Commands

```bash
conda create -n verl_plus python=3.11 && conda activate verl_plus
pip install -e .
pip install -r requirements.txt
pytest tests/test_protocol.py
pytest tests/ray_cpu
pre-commit run --all-files
```

Use the conda command for a README-compatible local environment. `pip install -e .` installs the package in editable mode, and `requirements.txt` adds dependencies. Run targeted pytest commands first; GPU, rollout, Ray, vLLM, and sglang tests may require CUDA, model weights, services, or extra packages.

## Coding Style & Naming Conventions

Python 3.11 is recommended for development, while package metadata allows Python >=3.8. Formatting and linting use Ruff via `.pre-commit-config.yaml`; run pre-commit before opening a PR. Ruff uses a 300-character line length and `verl` as the first-party import package. Use `snake_case` for modules, functions, variables, and shell scripts; use `PascalCase` for classes. Name experiment scripts descriptively, following patterns such as `Qwen2.5-1.5B_LogiQA_GRPO_only.sh`.

## Testing Guidelines

Write pytest tests as `test_*.py` under the relevant `tests/` subdirectory. Prefer CPU-only coverage for shared logic in `tests/utils/cpu_tests`, `tests/ray_cpu`, or focused package-level tests. Place GPU and distributed coverage in existing GPU/Ray/rollout folders and document hardware assumptions in PR notes. Run the smallest meaningful suite before broader checks.

## Commit & Pull Request Guidelines

Recent history uses short conventional-style prefixes such as `feat:`, `fix:`, and `doc:`. Keep commits focused and imperative, for example `fix: fol processing fallback`. PRs should summarize behavior changes, list commands run, link issues, and call out required datasets, model weights, checkpoints, or services. Include screenshots or logs only for visualization, UI, or experiment-output changes.

## Security & Configuration Tips

Do not commit credentials, `.netrc`, checkpoints, generated datasets, W&B runs, or local output directories. Keep environment-specific paths, model names, ports, and API endpoints configurable in scripts or YAML files, and document non-default requirements near the command that needs them.
