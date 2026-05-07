python examples/data_preprocess/mcq_preprocess.py \
    --input_parquet data/reclor/train.parquet \
    --base_url "http://localhost:4869/v1" \
    --model "qwen2.5-coder-7b-instruct" \
    --output_dir data/reclor \
    --verbose