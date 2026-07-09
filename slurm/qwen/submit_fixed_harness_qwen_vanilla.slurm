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
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/fixed_qwen_vanilla_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/fixed_qwen_vanilla_%j.err

# Fixed-harness eval: Qwen vanilla checkpoint (qwen_sweep_L24_27)

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
[ -f /home1/labiosa/.cache/huggingface/token ] && export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_ETAG_TIMEOUT=10

echo "=========================================="
echo "Fixed-harness: Qwen Vanilla (qwen_sweep_L24_27/lora_final)"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
RESULTS=$ROOT/models/qwen2.5/experiments/results
mkdir -p $RESULTS

$PY $ROOT/src/eval/eval_fixed_harness.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint $ROOT/stabilizer_weights/qwen_sweep_L24_27/lora_final \
    --n_samples 500 \
    --seed 42 \
    --output_file $RESULTS/fixed_harness_qwen_vanilla.json

echo "--- done ---"
