#!/bin/bash
#SBATCH --job-name=setup_gemma_env2
#SBATCH --account=robinjia_1822
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_gemma_env2_%j.out
#SBATCH --error=logs/setup_gemma_env2_%j.err

source ~/miniconda3/etc/profile.d/conda.sh

# Remove any partial env from previous attempt
conda env remove -n my_project_gemma -y 2>/dev/null || true

echo "Creating fresh env with Python 3.10..."
conda create -n my_project_gemma python=3.10 -y

conda activate my_project_gemma
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
PIP=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/pip

echo "Installing PyTorch..."
$PIP install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118 --quiet

echo "Installing transformers 4.44.2..."
$PIP install transformers==4.44.2 --quiet

echo "Installing peft, accelerate, bitsandbytes..."
$PIP install peft==0.11.0 accelerate==0.33.0 bitsandbytes==0.43.0 --quiet

echo "Installing data/utility packages..."
$PIP install datasets scipy scikit-learn tqdm sentencepiece numpy --quiet

echo "Verifying..."
$PY -c "
import torch, transformers, peft, accelerate
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('accelerate:', accelerate.__version__)
from transformers import AutoModelForCausalLM
print('Gemma2 support: OK')
"

echo "Done."
