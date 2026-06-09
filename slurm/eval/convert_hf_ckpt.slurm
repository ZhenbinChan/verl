#!/bin/bash
#SBATCH -J merge_ckpt
#SBATCH -o /home/chenzhenbin/Workspaces/verl/slurm/logs/eval/merge_ckpt.out
#SBATCH --qos=lv0b
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

module add cuda12.8/12.8

BASE_MODEL="/share/nlp/chenzhenbin/Workspaces/LLMs/Qwen3-8B-Base"
CKPT_DIR="/share/nlp/chenzhenbin/Workspaces/verl/ckpt/verl/qwen3-8b_logiqa_grpo_promptV1_format_v2/global_step_115/actor"
OUTPUT_DIR="/share/nlp/chenzhenbin/Workspaces/verl/output_models/qwen3-8b-base_warmup_115_v2"


python /share/nlp/chenzhenbin/Workspaces/verl/scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path ${BASE_MODEL} \
    --local_dir ${CKPT_DIR} \
    --target_dir ${OUTPUT_DIR} 

