#!/bin/bash
#SBATCH --job-name=setup_gemma_env
#SBATCH --account=robinjia_1822
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_gemma_env_%j.out
#SBATCH --error=logs/setup_gemma_env_%j.err

source ~/miniconda3/etc/profile.d/conda.sh

echo "Cloning my_project to my_project_gemma..."
conda create --clone my_project -n my_project_gemma -y

echo "Upgrading transformers to 4.44.2..."
conda activate my_project_gemma
pip install "transformers==4.44.2" --quiet

echo "Verifying..."
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "from transformers import AutoModelForCausalLM; print('AutoModel OK')"

echo "Done."
