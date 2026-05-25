#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v8_%j.out
#SBATCH --error=logs/train_v8_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- STARTING STABILIZER TRAINING (v8) ---"
nvidia-smi

mkdir -p logs

# v8 Architecture: Three fixes over v7
#
#   Fix 1 — Per-token gating (diagnostic fix):
#     The gate already outputs alpha per token position [B, T, 1] — architecture
#     unchanged. The diagnostic metric is now gate_max_ratio_s1 (ratio of EMA of
#     gate_s1_max on noisy steps vs clean steps) instead of gate_mean_ratio.
#     Target at step 3000: gate_max_ratio_s1 > 2.0.
#     Mean-based ratio diluted sparse perturbations (5% typo = 3/50 tokens);
#     max-based ratio captures whether any position is discriminating.
#
#   Fix 2 — fc2 init for stage 1:
#     Stage 1 fc2 now initialised with normal(std=0.01) instead of zeros.
#     Zero-init caused contrastive gradient dead zone: with delta=0 exactly,
#     d_corr = d_noisy and grad of L_contrastive w.r.t. fc2 = 0 from step 0.
#     Small random init breaks the symmetry so contrastive loss gets signal
#     immediately.  Stage 2 keeps zero-init (no contrastive loss there).
#
#   Fix 3 — dtype fix in register_inference_hooks():
#     Stabilizers stay in float32 at inference; hook casts input to float32,
#     output back to backbone dtype (float16).  Previous code cast stabilizer
#     to float16, which caused RuntimeError in CUDA LayerNorm during milestone
#     eval.
#
#   Fix 4 — Increased lambda_suppress (0.35 vs 0.20):
#     v6 multidomain showed -9% clean regression with gate_max ≈ 0.39.
#     Stronger clean-step gate suppression penalty to prevent gate from
#     opening on unperturbed inputs in the broader multidomain distribution.
#
# KEY DIAGNOSTIC: Check gate_max_ratio_s1 at step 3000.
#   Target: > 2.0  (embedding gate max discriminating between clean and noisy)
#   Failure: ~1.0  (even max gate not discriminating — abort and revisit)
#
# Training data: 40% GSM8K / 30% MMLU / 30% C4 (multidomain)
# Evaluation: GSM8K-only (comparable to v1-v7)
python src/lib/train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  mixed \
    --output_dir ./stabilizer_weights/phi35_mixed_mlp_v8 \
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
