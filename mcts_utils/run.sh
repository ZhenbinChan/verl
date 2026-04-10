export OPENAI_BASE_URL="https://litellm.mybigai.ac.cn/"
export OPENAI_API_KEY="sk-1Eobh8xPXHLolQZW4wZA_w"
export CUDA_VISIBLE_DEVICES=0

python mcts_utils/main.py \
    --tokenizer_path "/home/linziyong/Desktop/Model/Qwen2.5-3B-Instruct" \
    --eval_path "data/logiqa.jsonl" \
    --batch_size 16 \
    --generation_strategy "mcts" \
    --evaluation_strategy "nli" \
    --random_pick