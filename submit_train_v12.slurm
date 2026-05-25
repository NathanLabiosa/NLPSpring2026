#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v12_%j.out
#SBATCH --error=logs/train_v12_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- STARTING STABILIZER TRAINING (v12) ---"
nvidia-smi

mkdir -p logs

# v12 Architecture: hard-freeze stage 1 gate to α=1.0 during warmup
#
# Root cause of v9–v11 gate collapse:
#   fc2 is random at init → delta is random → embed_mse gradient to α is
#   incoherent (zero in expectation).  Even tiny regularization (λ_suppress=0.02)
#   wins and collapses α to 0 within 500 steps.  Once α→0, ∂L/∂fc2 ∝ α→0 too,
#   so neither fc2 nor the gate can recover.  This is a chicken-and-egg deadlock.
#
# Fix (v12): hard-fix α=1.0 for stage 1 for the first gate_warmup_steps steps.
#   gate_net is bypassed entirely (no gradient through it).  fc2 receives the
#   full embed_mse gradient (∂L/∂fc2 ∝ α=1.0) from step 0 and quickly learns
#   delta ≈ h_clean − h_noisy.  L_suppress / L_gate / L_adapter are all zero
#   during this phase (no point penalizing a fixed gate).
#
#   At step gate_warmup_steps (3000): gate unfreezes.  delta is already coherent,
#   so ∂L_mse/∂α = 2*delta·(h_corr−h_clean) < 0 on noisy examples (gate gets
#   a strong opening signal immediately).  suppress/gate/adapter then ramp from
#   0 → full over reg_ramp_end_steps (3000) to teach gate selectivity.
#
# KEY DIAGNOSTICS at step 3000 (unfreeze):
#   emb_mse should be LOW (fc2 has learned) — expect < 0.001 vs 0.002 at step 0
#   s1_mx should JUMP above 0.27 immediately after unfreeze (gate gets coherent signal)
#   r1_max should grow > 1.5 by step 6000 (gate discriminates noisy vs clean)
#   gnorm should be non-zero throughout warmup (fc2 is actively training)
#
python train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  mixed \
    --output_dir ./stabilizer_weights/phi35_mixed_mlp_v12 \
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
    --lambda_gate_start  0.0 \
    --lambda_suppress    0.35 \
    --lambda_suppress_start 0.0 \
    --lambda_adapter     0.15 \
    --probe_layers 9 \
    --ema_decay 0.99 \
    --gate_warmup_steps  3000 \
    --reg_ramp_end_steps 3000 \
    --stab_warmup_frac   0.25 \
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
