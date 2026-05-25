#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --output=logs/download_gemma2_9b_%j.out
#SBATCH --error=logs/download_gemma2_9b_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "--- Removing gemma-2-9b-it cache ---"
rm -rf ~/.cache/huggingface/hub/models--google--gemma-2-9b-it
echo "Done removing instruct cache."

echo "--- Downloading google/gemma-2-9b ---"
/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_id = "google/gemma-2-9b"
token = os.environ["HF_TOKEN"]

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(model_id, token=token)
print("Downloading model weights...")
AutoModelForCausalLM.from_pretrained(model_id, token=token)
print("Download complete.")
EOF
