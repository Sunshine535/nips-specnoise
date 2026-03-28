#!/usr/bin/env bash
set -euo pipefail

# Activate venv if available
_PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_PROJ_ROOT/.venv/bin/activate" ]; then source "$_PROJ_ROOT/.venv/bin/activate"; fi
export PATH="$HOME/.local/bin:$PATH"

# ─── SpecNoise Search Launcher (8x A100-80GB) ───
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export WANDB_PROJECT="specnoise"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/noise_grid.yaml"
OUTPUT_DIR="${PROJECT_DIR}/results/noise_search"

mkdir -p "$OUTPUT_DIR"

NUM_GPUS=8
MASTER_PORT="${MASTER_PORT:-29503}"

echo "========================================="
echo " SpecNoise Search"
echo " GPUs: ${NUM_GPUS}"
echo " Config: ${CONFIG}"
echo " Output: ${OUTPUT_DIR}"
echo "========================================="

python "${SCRIPT_DIR}/noise_search.py" \
    --config_path "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "Noise search complete."
echo "Run noise_guided_sft.py for Stage 2 and eval_noise_specialization.py for evaluation."
