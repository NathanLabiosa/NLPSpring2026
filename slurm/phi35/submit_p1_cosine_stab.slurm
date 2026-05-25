#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/p1_cosine_stab_%j.out
#SBATCH --error=logs/p1_cosine_stab_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- P1: Layers 0–4, perturbed, cosine L_stab ---"
nvidia-smi
mkdir -p logs

# v17 variant: L_stab uses cosine distance instead of MSE.
# Tests whether the paper's own diagnostic metric (cosine-based LRD)
# works as a training signal. Identical to v17 except --stab_cosine.
#
# MSE penalizes large-magnitude shifts even when direction is preserved.
# Cosine distance penalizes directional change regardless of magnitude.
# If cosine L_stab outperforms MSE L_stab, it tightens the narrative:
# "we diagnose with cosine LRD, and regularize with the same metric."

python src/training/train_lrd_lora_v14.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/p1_cosine_stab \
    --lora_rank    4 \
    --lora_alpha   8 \
    --lora_dropout 0.05 \
    --lora_layers  5 \
    --target_modules qkv_proj o_proj \
    --lambda_acc   0.7 \
    --lambda_stab  0.3 \
    --stab_layer   9 \
    --stab_cosine \
    --n_per_condition 857 \
    --clean_fraction  0.25 \
    --epochs 30 \
    --max_steps 300 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 25 \
    --save_every 300 \
    --eval_every 0 \
    --clean_eval_every 300 \
    --early_stop_delta 5.0 \
    --clean_eval_n 500 \
    --eval_after_training \
    --eval_n_samples 500
