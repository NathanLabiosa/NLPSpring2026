#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v10_%j.out
#SBATCH --error=logs/train_v10_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- STARTING STABILIZER TRAINING (v10) ---"
nvidia-smi

mkdir -p logs

# v10 Architecture: Fix embed_mse_loss — changed-token mask instead of mean-pooling
#
#   v9 failure: mean-pooling MSE over ~50-token prompt dilutes the signal by
#     (n_changed/T)^2 ≈ (3/50)^2.  With ~3 changed tokens out of 50, the
#     mean MSE was ~0.0001, 70x weaker than lambda_suppress * L_suppress.
#     Gate collapsed monotonically from 0.05 to 0.0000 by step 500.
#
#   v10 fix: position-wise MSE restricted to positions where token IDs differ.
#
#     changed_mask = (clean_ids[:, :T_min] != noisy_ids[:, :T_min])
#                    & clean_attn_mask[:, :T_min]   # exclude left-padding
#                    & noisy_attn_mask[:, :T_min]
#     L_embed_mse  = per_pos_mse[changed_mask].mean()
#
#   Properties:
#     • Full per-token signal at genuinely perturbed positions (~3 tokens/example
#       at 5% typo rate → emb_mse ≈ 0.02, comparable to suppress signal).
#     • Zero gradient at unchanged tokens — gate correctly learns to stay closed
#       there without any extra regularization.
#     • Padding positions excluded by attention-mask intersection.
#     • Token-split misalignment (typo "colour"→["c","0","l","0","ur"] shifts
#       positions after the split): those positions are in changed_mask but have
#       noisy gradient targets.  They average out over training; correctly-aligned
#       changed positions dominate.  Accepted imperfection — far better than
#       mean-pooled signal or penalizing all positions.
#
#   All other hyperparameters unchanged from v9:
#     max_norm=0.05 (fraction of ‖h‖), lambda_suppress=0.35, dataset=mixed,
#     stage 2 config, curriculum, etc.
#
# KEY DIAGNOSTIC: Check emb_mse at step 3000.
#   Target: emb_mse in range 0.01–0.10 and DECREASING (confirms changed-mask
#           signal is coming through, not diluted to 0.0001 as in v9)
#   Target: r1_max > 1.3  (gate opening selectively on noisy steps)
#   Target: gnorm > 0 consistently on noisy steps
#   Failure: emb_mse still ≈ 0.0001 → check that IDs are being passed correctly
#   Failure: gate collapse again → reduce lambda_suppress (0.35 → 0.20)
#
# Training data: 40% GSM8K / 30% MMLU / 30% C4 (multidomain)
# Evaluation: GSM8K-only (comparable to v1-v9)
python src/lib/train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  mixed \
    --output_dir ./stabilizer_weights/phi35_mixed_mlp_v10 \
    --inject_layers 2 4 \
    --bottleneck_dim 128 \
    --stage2_bottleneck_dim 32 \
    --max_norm 0.05 \
    --gate_dim 64 \
    --lambda_embed_mse   0.35 \
    --lambda_stab        0.15 \
    --lambda_acc         0.40 \
    --lambda_prop        0.10 \
    --lambda_gate        0.15 \
    --lambda_suppress    0.35 \
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
