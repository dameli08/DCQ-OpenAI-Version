#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# Step 2: Start vLLM server (non-thinking mode)
# Keep this terminal open while running contamination detection.
# Change ONLY this variable:
MODEL_PATH="/data/models/Qwen3.5-2B"
# ════════════════════════════════════════════════════════════════════

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dcq

CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 vllm serve ${MODEL_PATH} \
    --port 23333 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager
