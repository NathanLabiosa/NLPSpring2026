#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/Gemma2/logs/eval_L30_35_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/Gemma2/logs/eval_L30_35_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "--- Gemma-2-9B Eval L30-35 ---"
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
MODEL=google/gemma-2-9b
SW=$ROOT/stabilizer_weights

$PY $ROOT/eval_only_lora_window.py \
    --model $MODEL --output_dir $SW/gemma2_9b_sweep_L30_35 \
    --eval_batch_size 8 --n_eval 500

echo "--- L30-35 eval complete ---"
