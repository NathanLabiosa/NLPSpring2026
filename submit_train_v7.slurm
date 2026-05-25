#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v7_%j.out
#SBATCH --error=logs/train_v7_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- STARTING STABILIZER TRAINING (v7 two-stage) ---"
nvidia-smi

mkdir -p logs

# v7 Architecture: Two-stage redesign
#   Stage 1: Embedding-level correction (before layer 0)
#     - Gate reads concat([emb, |emb - emb_clean_mean|]) — EMA always active
#     - Contrastive loss: max(0, d(corrected,clean) - d(noisy,clean) + 0.1)
#     - Bottleneck 128
#   Stage 2: Refinement at layers 2 and 4
#     - Gate reads h_norm only (no EMA aug)
#     - MSE reconstruction loss
#     - Smaller bottleneck 32
#
# Curriculum (3-phase):
#   Steps 0–2000:   reg_mult=2.0 (heavy regularization, gate stays closed)
#   Steps 2000–8000: reg_mult decays 2.0→1.0
#   Steps 8000+:    reg_mult=1.0 (gate free to open)
#
# KEY DIAGNOSTIC: Check gate_ratio_s1 at step 3000.
#   Target: > 1.3  (embedding gate discriminating between clean and noisy)
#   Failure: ~1.0  (embedding signal not sufficient — abort and revisit)
#
# Training data: 40% GSM8K / 30% MMLU / 30% C4 (multidomain)
# Evaluation: GSM8K-only (comparable to v1-v6)
python train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  mixed \
    --output_dir ./stabilizer_weights/phi35_mixed_mlp_v7 \
    --inject_layers 2 4 \
    --bottleneck_dim 128 \
    --stage2_bottleneck_dim 32 \
    --max_norm 1.0 \
    --gate_dim 64 \
    --lambda_contrast    0.35 \
    --contrastive_margin 0.10 \
    --lambda_stab        0.15 \
    --lambda_acc         0.40 \
    --lambda_prop        0.10 \
    --lambda_gate        0.15 \
    --lambda_suppress    0.20 \
    --lambda_adapter     0.15 \
    --probe_layers 9 \
    --ema_decay 0.99 \
    --reg_warmup_steps    2000 \
    --reg_decay_end_steps 8000 \
    --stab_warmup_frac    0.25 \
    --epochs 20 \
    --max_steps 25000 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --n_per_condition 500 \
    --max_seq_len 512 \
    --log_every 25 \
    --save_every 500 \
    --eval_after_training \
    --eval_n_samples 100
