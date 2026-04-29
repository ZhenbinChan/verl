#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

PROMPT_FILE=${PROMPT_FILE:-logical_reasoning.txt}
DATA_DIR=${DATA_DIR:-"$REPO_ROOT/data/logiqa2k_prompt_v2"}
VERSION=${VERSION:-1}
NUM_SAMPLES=${NUM_SAMPLES:--1}
FORMAT=${FORMAT:-xml}

python "$REPO_ROOT/examples/data_preprocess/logiqa.py" \
    --version "$VERSION" \
    --num_samples "$NUM_SAMPLES" \
    --format "$FORMAT" \
    --local_save_dir "$DATA_DIR" \
    --system_prompt_file "$PROMPT_FILE"

echo "Wrote prompt-v2 parquet files to: $DATA_DIR"
echo "Prompt file: $PROMPT_FILE"
