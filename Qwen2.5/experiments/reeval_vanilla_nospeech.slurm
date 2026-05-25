#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/qwen_reeval_vanilla_nospeech_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/qwen_reeval_vanilla_nospeech_%j.err

# Re-evaluate vanilla_nospeech with fixed max_new_tokens=512

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project/bin/python
SW=$ROOT/stabilizer_weights

# Use the standard eval_only_lora_window.py which has correct max_new_tokens=512
$PY $ROOT/eval_only_lora_window.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output_dir $SW/qwen_vanilla_nospeech \
    --eval_batch_size 8 \
    --n_eval 500

echo "--- Re-evaluation complete ---"
