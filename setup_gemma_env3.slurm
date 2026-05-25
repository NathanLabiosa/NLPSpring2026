#!/bin/bash
#SBATCH --job-name=setup_gemma_env3
#SBATCH --account=robinjia_1822
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup_gemma_env3_%j.out
#SBATCH --error=logs/setup_gemma_env3_%j.err

# Fast approach: venv that inherits my_project packages via --system-site-packages
# Only need to pip-install the transformers upgrade on top — no file copying.

BASE_PY=/home1/labiosa/miniconda3/envs/my_project/bin/python
VENV=/home1/labiosa/miniconda3/envs/my_project_gemma

echo "Removing any partial previous env..."
rm -rf $VENV

echo "Creating venv with system-site-packages (inherits my_project)..."
$BASE_PY -m venv --system-site-packages $VENV

PIP=$VENV/bin/pip
PY=$VENV/bin/python

echo "Upgrading transformers to 4.44.2..."
$PIP install "transformers==4.44.2" --quiet

echo "Verifying..."
$PY -c "
import torch, transformers, peft
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
from transformers import AutoConfig
cfg = AutoConfig.for_model('gemma2')
print('Gemma2 config support: OK')
"

echo "Done."
