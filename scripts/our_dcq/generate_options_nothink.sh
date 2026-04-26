#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# Step 1: Generate paraphrase options via OpenAI (non-thinking mode)
# Change ONLY these two variables (or pass as env vars):
DATASET="${DATASET:-MMLU_KAZ_Translation}"
MODEL_NAME="${MODEL_NAME:-qwen3_5_9b}"
# ════════════════════════════════════════════════════════════════════

DATASETS_DIR="/home/dameli/DCQ-OpenAI-Version/8_datasets"
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
SAMPLE_FILE="${DATASETS_DIR}/${DATASET}_100_${MODEL_NAME}.csv"
PROCESSED_DIR="${DATASETS_DIR}/processed/${MODEL_NAME}/${DATASET_LOWER}_100"
OPENAI_MODEL="gpt-5-mini"

cd /home/dameli/DCQ-OpenAI-Version
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dcq

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Run: export OPENAI_API_KEY=your_key"
    exit 1
fi

# Sample 100 rows from the dataset
python - <<PY
import pandas as pd
df = pd.read_csv("${DATASETS_DIR}/${DATASET}.csv", encoding="utf-8")
sample = df.sample(n=100)
sample.to_csv("${SAMPLE_FILE}", index=False, encoding="utf-8")
print("Saved:", "${SAMPLE_FILE}", " rows:", len(sample))
PY

# Generate paraphrases with OpenAI
python src/generating_quiz_options.py \
    --filepath "${SAMPLE_FILE}" \
    --processed_dir "${PROCESSED_DIR}" \
    --columns_to_form_instances "question" \
    --model "${OPENAI_MODEL}"