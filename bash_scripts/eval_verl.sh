set -x

HOME=/home/chenzhb/Workspaces/verl

MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct

DATA_PATH=/home/chenzhb/Workspaces/verl/data/logiqa_tree/test.parquet

python3 -m verl.trainer.main_eval \
    data.path=$DATA_PATH \
    data.prompt_key='prompt' \
    data.response_key='response' \
    data.data_source_key='data_source' \
    data.reward_model_key='reward_model' $@