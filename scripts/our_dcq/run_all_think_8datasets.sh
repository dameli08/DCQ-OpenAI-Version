#!/bin/bash
set -euo pipefail

# Configure once
MODEL_NAME="${MODEL_NAME:-qwen3_5_2b}"
RESULTS_MODEL_NAME="${RESULTS_MODEL_NAME:-${MODEL_NAME}_think}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3.5-2B}"

DATASETS=(
  mmlu_all
  mmlu_pro_all
  #mmlu_cf_all
  #kazmmlu_all
  #rummlu_all
  #MMLU_RUS_Translation
  #MMLU_KAZ_Translation
)

DATASETS_DIR="/home/dameli/DCQ-OpenAI-Version/8_datasets"

cd /home/dameli/DCQ-OpenAI-Version

# vLLM should already be running before this script starts.
if ! curl -s http://localhost:23333/health >/dev/null 2>&1; then
  echo "ERROR: vLLM is not reachable at http://localhost:23333"
  echo "Start your Qwen vLLM server first, then rerun."
  exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dcq

for DATASET in "${DATASETS[@]}"; do
  DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
  PROCESSED_FILE="${DATASETS_DIR}/processed/${MODEL_NAME}/${DATASET_LOWER}_100/${DATASET}_100_${MODEL_NAME}.csv"
  EXPERIMENT_DIR="/home/dameli/DCQ-OpenAI-Version/results/our_dcq/wild_evaluation/${RESULTS_MODEL_NAME}/${DATASET_LOWER}_100"

  echo "============================================================"
  echo "Starting dataset: ${DATASET}"
  echo "MODEL_NAME=${MODEL_NAME}"
  echo "RESULTS_MODEL_NAME=${RESULTS_MODEL_NAME}"
  echo "============================================================"

  OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://localhost:23333/v1 \
  python src/taking_quiz.py \
    --filepath "${PROCESSED_FILE}" \
    --dataset "${DATASET}" \
    --split test \
    --experiment "${EXPERIMENT_DIR}" \
    --model "${MODEL_PATH}" \
    --dataset_instances_column_name "instance" \
    --thinking

  echo ""
  echo "============================================================"
  echo "RESULT REPORT: ${DATASET}"
  echo "============================================================"
  cat "${EXPERIMENT_DIR}/data_contamination_report_for_${DATASET}_100_${MODEL_NAME}.txt"
done

echo "============================================================"
echo "Done. All thinking-mode datasets completed."
echo "============================================================"
