#!/bin/bash
set -euo pipefail

# Configure once
MODEL_NAME="${MODEL_NAME:-qwen3_5_2b}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3.5-2B}"

DATASETS=(
  # mmlu_all
  # mmlu_pro_all
  mmlu_cf_all
  # mmlu_redux_all
  kazmmlu_all
  rummlu_all
  MMLU_RUS_Translation
  MMLU_KAZ_Translation
)

cd /home/dameli/DCQ-OpenAI-Version

# vLLM should already be running before this script starts.
if ! curl -s http://localhost:23333/health >/dev/null 2>&1; then
  echo "ERROR: vLLM is not reachable at http://localhost:23333"
  echo "Start your qwen3.5-9B vLLM server first, then rerun."
  exit 1
fi

for DATASET in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "Starting dataset: ${DATASET}"
  echo "MODEL_NAME=${MODEL_NAME}"
  echo "============================================================"

  DATASET="${DATASET}" MODEL_NAME="${MODEL_NAME}" \
    bash scripts/our_dcq/generate_options_nothink.sh

  DATASET="${DATASET}" MODEL_NAME="${MODEL_NAME}" MODEL_PATH="${MODEL_PATH}" \
    bash scripts/our_dcq/run_contamination_nothink.sh

done

echo "============================================================"
echo "Done. All 8 datasets completed."
echo "============================================================"
