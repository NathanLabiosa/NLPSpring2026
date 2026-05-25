#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=14:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/Qwen2.5/logs/eval_L24_27_stab_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/Qwen2.5/logs/eval_L24_27_stab_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "--- Qwen2.5-7B Stabilizer Eval L24_27 ---"
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
export PYTHONPATH=$ROOT/scripts:$PYTHONPATH
PY=/home1/labiosa/miniconda3/envs/my_project/bin/python
MODEL=Qwen/Qwen2.5-7B-Instruct
SW=$ROOT/stabilizer_weights

$PY $ROOT/eval_only_lora_window.py \
    --model $MODEL --output_dir $SW/qwen25_7b_L24_27_stab \
    --eval_batch_size 8 --n_eval 500

echo "--- L24_27 stabilizer eval complete ---"
