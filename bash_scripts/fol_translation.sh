#!/bin/bash

python3 examples/data_preprocess/logiqa_fol_preprocess.py \
    --input_parquet data/logiqa_action_1k/train.parquet \
    --local_dir data/logiqa_fol \
    --api_key 'empty' \
    --skip_output_parquet