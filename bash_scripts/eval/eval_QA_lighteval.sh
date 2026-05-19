export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
MODEL=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct

# MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.0,top_p:1}"

OUTPUT_DIR=/home/chenzhb/Workspaces/verl/eval_output/Qwen2.5-1.5B-Instruct



# ============================================================
# In-domain Evaluation
# ============================================================

# lucasmccabe/logiqa
lighteval vllm $MODEL_ARGS "community|logiqa|0|0" \
  --custom-tasks /home/chenzhb/Workspaces/lighteval/community_tasks/logiqa_task.py \
  --use-chat-template \
  --output-dir $OUTPUT_DIR

# ReClor
lighteval vllm $MODEL_ARGS "community|reclor|0|0" \
  --custom-tasks /home/chenzhb/Workspaces/lighteval/community_tasks/reclor_task.py \
  --use-chat-template \
  --output-dir $OUTPUT_DIR

# AR-LSAT
lighteval vllm $MODEL_ARGS "community|arlsat|0|0" \
  --custom-tasks /home/chenzhb/Workspaces/lighteval/community_tasks/arlsat_task.py \
  --use-chat-template \
  --output-dir $OUTPUT_DIR

# ============================================================
# Knowledge / Factual Recall
# ============================================================


# TriviaQA
lighteval vllm $MODEL_ARGS "lighteval|triviaqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# ============================================================
# Commonsense / Logical / Multi-step Reasoning
# ============================================================


# LSAT QA — logical/analytical reasoning
lighteval vllm $MODEL_ARGS "helm|lsat_qa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# OpenBookQA (HELM)
lighteval vllm $MODEL_ARGS "helm|openbookqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# OpenBookQA (lighteval)
lighteval vllm $MODEL_ARGS "lighteval|openbookqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Main — expert-written, Google-proof deep reasoning
lighteval vllm $MODEL_ARGS "lighteval|gpqa:main|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MathQA — math word problem reasoning
lighteval vllm $MODEL_ARGS "lighteval|mathqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# QA4MRE 2013 — machine reading comprehension + reasoning
lighteval vllm $MODEL_ARGS "lighteval|qa4mre:2013|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# ============================================================
# Truthfulness / Factuality Reasoning
# ============================================================

# TruthfulQA MC — distinguishing facts from misconceptions
lighteval vllm $MODEL_ARGS "leaderboard|truthfulqa:mc|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


# ============================================================
# Domain-specific Reasoning (Medical / Biomedical)
# ============================================================

# MedMCQA — medical multiple choice
lighteval vllm $MODEL_ARGS "helm|med_mcqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# PubMedQA (HELM)
lighteval vllm $MODEL_ARGS "helm|pubmedqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# PubMedQA (lighteval)
lighteval vllm $MODEL_ARGS "lighteval|pubmedqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# Strategyqa
lighteval vllm $MODEL_ARGS "bigbench|strategyqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# Strategyqa
lighteval vllm $MODEL_ARGS "lighteval|triviaqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# ============================================================
# Aggregate all results into a summary table
# ============================================================
echo ""
echo "============================================================"
echo "  Aggregating QA evaluation results..."
echo "============================================================"
bash $(dirname "$0")/aggregate_qa_results.sh $OUTPUT_DIR
