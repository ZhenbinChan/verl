python examples/data_preprocess/mcq_preprocess.py \
    --input_parquet data/reclor/train.parquet \
    --base_url "http://localhost:4869/v1" \
    --model "qwen2.5-7b-coder" \
    --output_dir data/reclor_fol \
    --max_retries -1 \
    --verbose