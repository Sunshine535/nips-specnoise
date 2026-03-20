#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="specnoise"

echo "============================================"
echo " SpecNoise: Weight Noise for Domain Specialization in LLMs — Environment Setup"
echo "============================================"

# Create conda env
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME" || source activate "$ENV_NAME"

# Install PyTorch
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install project deps
pip install -r "$PROJ_DIR/requirements.txt"

# Install flash-attention (optional, skip on error)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn not installed (optional)"

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.device_count()} GPUs')"

echo "Setup complete! Run: bash scripts/run_all_experiments.sh"
