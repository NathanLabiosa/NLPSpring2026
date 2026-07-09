#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/models/qwen2.5/logs/train_L12_15_stab_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/models/qwen2.5/logs/train_L12_15_stab_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "--- Qwen2.5-7B L12_15 with L_stab ---"
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
export PYTHONPATH=$ROOT/scripts:$PYTHONPATH
PY=/home1/labiosa/miniconda3/envs/my_project/bin/python
MODEL=Qwen/Qwen2.5-7B-Instruct
SW=$ROOT/stabilizer_weights

$PY $ROOT/train_lrd_lora_v14.py \
    --model $MODEL \
    --dataset gsm8k \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_layers_start 12 \
    --lora_layers 4 \
    --stab_layer 20 \
    --target_modules q_proj v_proj \
    --lambda_acc 0.7 \
    --lambda_stab 3.0 \
    --stab_cosine \
    --n_per_condition 857 \
    --clean_fraction 0.25 \
    --epochs 30 \
    --max_steps 1000 \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --lr 2e-5 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 50 \
    --save_every 1000 \
    --eval_every 0 \
    --clean_eval_every 200 \
    --early_stop_delta 15.0 \
    --clean_eval_n 200 \
    --eval_n_samples 500 \
    --output_dir $SW/qwen25_7b_L12_15_stab

echo "--- L12_15 complete ---"
