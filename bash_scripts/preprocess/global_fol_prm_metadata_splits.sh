#!/bin/bash

python3 examples/data_preprocess/global_fol_prm_metadata_splits.py \
    --input_dir data/reclor \
    --output_dir data/reclor_global_fol_prm \
    --dataset_namespace reclor \
    --api_config llm_server/configs/minimax.yaml \
    --max_workers 4 \
    --max_retries 3 \
    --save_every 100 \
    "$@"
