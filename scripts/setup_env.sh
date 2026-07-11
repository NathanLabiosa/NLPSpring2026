#!/usr/bin/env bash
# setup_env.sh — create the nlp2026 conda environment for this project.
# Run once: bash scripts/setup_env.sh
# Takes ~10 minutes (torch download is large).

set -euo pipefail

ENV_NAME="nlp2026"
CONDA_BASE="/home/nathanlabiosa/miniforge3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================"
echo "Creating conda env: $ENV_NAME"
echo "Repo root: $ROOT"
echo "============================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create env with Python 3.11
conda create -n "$ENV_NAME" python=3.11 -y

PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

# Install PyTorch (CUDA 12.6, matching the cluster)
"$PIP" install torch==2.7.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Install NLP stack (pin to versions tested on lerobot_smol env)
"$PIP" install \
    "transformers==4.53.3" \
    "peft==0.17.1" \
    "datasets==4.1.1" \
    "accelerate>=0.26.0" \
    "sentencepiece" \
    "tqdm" \
    "huggingface_hub" \
    "scipy" \
    "numpy"

PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
echo ""
echo "============================================"
echo "Env ready: $ENV_NAME"
echo "Python:    $PY"
echo ""
"$PY" -c "import torch,transformers,peft; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); print('ALL OK')"
echo "============================================"
