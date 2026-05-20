#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/chenzhb/Workspaces/verl}
INPUT_PARQUET=${INPUT_PARQUET:-${PROJECT_ROOT}/data/logiqa/train.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/data/logiqa_global_fol_prm}

FOL_PROVIDER=${FOL_PROVIDER:-minimax}
FOL_BASE_URL=${FOL_BASE_URL:-https://api.minimaxi.com/v1}
FOL_MODEL=${FOL_MODEL:-MiniMax-M2.7}
FOL_AZURE_ENDPOINT=${FOL_AZURE_ENDPOINT:-}
FOL_API_VERSION=${FOL_API_VERSION:-}
FOL_DEPLOYMENT_NAME=${FOL_DEPLOYMENT_NAME:-}

MAX_WORKERS=${MAX_WORKERS:-4}
MAX_RETRIES=${MAX_RETRIES:-3}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-60}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-0.1}
TOP_P=${TOP_P:-0.8}
SAVE_EVERY=${SAVE_EVERY:-100}
NUM_SAMPLES=${NUM_SAMPLES:-}

cd "${PROJECT_ROOT}"

args=(
  examples/data_preprocess/global_fol_prm_metadata.py
  --input_parquet "${INPUT_PARQUET}"
  --output_dir "${OUTPUT_DIR}"
  --provider "${FOL_PROVIDER}"
  --base_url "${FOL_BASE_URL}"
  --model "${FOL_MODEL}"
  --max_workers "${MAX_WORKERS}"
  --max_retries "${MAX_RETRIES}"
  --request_timeout "${REQUEST_TIMEOUT}"
  --max_tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --save_every "${SAVE_EVERY}"
)

if [[ -n "${FOL_API_KEY:-}" ]]; then
  args+=(--api_key "${FOL_API_KEY}")
fi
if [[ -n "${FOL_AZURE_ENDPOINT}" ]]; then
  args+=(--azure_endpoint "${FOL_AZURE_ENDPOINT}")
fi
if [[ -n "${FOL_API_VERSION}" ]]; then
  args+=(--api_version "${FOL_API_VERSION}")
fi
if [[ -n "${FOL_DEPLOYMENT_NAME}" ]]; then
  args+=(--deployment_name "${FOL_DEPLOYMENT_NAME}")
fi
if [[ -n "${NUM_SAMPLES}" ]]; then
  args+=(--num_samples "${NUM_SAMPLES}")
fi
if [[ "${SAVE_RAW_RESPONSE:-0}" == "1" ]]; then
  args+=(--save_raw_response)
fi

python3 "${args[@]}"
