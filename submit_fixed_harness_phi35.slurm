#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/fixed_phi35_stab_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/fixed_phi35_stab_%j.err

# Fixed-harness eval: Phi-3.5 stabilizer checkpoint (cosv2_L15_stab24_high)

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
[ -f /home1/labiosa/.cache/huggingface/token ] && export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_ETAG_TIMEOUT=10

echo "=========================================="
echo "Fixed-harness: Phi-3.5 Stabilizer (cosv2_L15_stab24_high/lora_final)"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
RESULTS=$ROOT/Qwen2.5/experiments/results
mkdir -p $RESULTS

$PY $ROOT/eval_fixed_harness.py \
    --base_model microsoft/Phi-3.5-mini-instruct \
    --checkpoint $ROOT/stabilizer_weights/cosv2_L15_stab24_high/lora_final \
    --n_samples 500 \
    --seed 42 \
    --attn_impl eager \
    --output_file $RESULTS/fixed_harness_phi35_stab.json

echo "--- done ---"
