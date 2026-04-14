#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# Step 3: Run contamination detection (non-thinking mode)
# Run this in a second terminal while vLLM server is running.
# Change ONLY these three variables (or pass as env vars):
DATASET="${DATASET:-MMLU_KAZ_Translation}"
MODEL_NAME="${MODEL_NAME:-qwen3_5_9b}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3.5-9B}"
# ════════════════════════════════════════════════════════════════════

# Auto-derived (do not change)
DATASETS_DIR="/home/dameli/DCQ-OpenAI-Version/8_datasets"
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
PROCESSED_FILE="${DATASETS_DIR}/processed/${MODEL_NAME}/${DATASET_LOWER}_100/${DATASET}_100_${MODEL_NAME}.csv"
EXPERIMENT_DIR="/home/dameli/DCQ-OpenAI-Version/results/our_dcq/wild_evaluation/${MODEL_NAME}/${DATASET_LOWER}_100"

cd /home/dameli/DCQ-OpenAI-Version
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dcq

OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://localhost:23333/v1 \
python src/taking_quiz.py \
    --filepath "${PROCESSED_FILE}" \
    --dataset "${DATASET}" \
    --split test \
    --experiment "${EXPERIMENT_DIR}" \
    --model "${MODEL_PATH}" \
    --dataset_instances_column_name "instance"

# Show result report
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "RESULT REPORT:"
echo "════════════════════════════════════════════════════════════════"
cat "${EXPERIMENT_DIR}/data_contamination_report_for_${DATASET}_100_${MODEL_NAME}.txt"
