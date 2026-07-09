#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/models/gemma2/logs/train_L24_29_stab_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/models/gemma2/logs/train_L24_29_stab_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/openmpi-5.0.5-mufqd73/lib:/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/openblas-0.3.28-mgrljin/lib64:/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/python-3.11.9-x74mtjf/lib:/apps/generic/gcc/13.3.0/lib64
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "--- Gemma-2-9B L24_29 with L_stab ---"
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
MODEL=google/gemma-2-9b
SW=$ROOT/stabilizer_weights

$PY $ROOT/train_lrd_lora_v14.py \
    --model $MODEL \
    --dataset gsm8k \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_layers_start 24 \
    --lora_layers 6 \
    --stab_layer 34 \
    --target_modules q_proj v_proj \
    --lambda_acc 0.7 \
    --lambda_stab 3.0 \
    --stab_cosine \
    --n_per_condition 857 \
    --clean_fraction 0.25 \
    --epochs 30 \
    --max_steps 1000 \
    --batch_size 2 \
    --grad_accum_steps 8 \
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
    --output_dir $SW/gemma2_9b_L24_29_stab

echo "--- L24_29 complete ---"
