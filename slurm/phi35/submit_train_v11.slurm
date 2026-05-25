#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v11_%j.out
#SBATCH --error=logs/train_v11_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- STARTING STABILIZER TRAINING (v11) ---"
nvidia-smi

mkdir -p logs

# v11 Architecture: Two fixes for gate collapse
#
#   v10 failure: gate collapses from α=0.05 to 0.0000 by step 500.
#     Root causes:
#     (1) Gate bias -3.0 → α_init=0.05. At low α, sigmoid gradient ≈ 0.047.
#         fc2 random at init → delta·(h_noisy-h_clean) ≈ 0 (incoherent) → MSE
#         gradient to gate is zero even though emb_mse>0. Gate can't open.
#     (2) reg_mult=2.0 at start meant suppress was 2×0.35=0.70 during the first
#         2000 steps — 10× stronger closing force than the MSE opening signal.
#         Gate was dead on arrival.
#
#   Fix 1 — Warm-start gate bias: -3.0 → -1.0 (α_init: 0.05 → 0.27).
#     At α=0.27, sigmoid gradient = 0.197 (4.2× larger than at 0.05).
#     fc2 gets a real gradient from step 0, allowing it to learn which
#     directions reduce emb_mse before regularization drives the gate down.
#
#   Fix 2 — Invert the reg schedule for suppress/gate: start LOW, ramp UP.
#     Old schedule: suppress=2×0.35=0.70 at step 0, decays to 0.35 by step 8000.
#     New schedule: suppress starts at 0.02, ramps to 0.35 by step 3000.
#                   gate starts at 0.02, ramps to 0.15 by step 3000.
#     This gives the opening signal (emb_mse) a 3000-step head start to build
#     a coherent gradient direction before suppression kicks in at full strength.
#     After step 3000, full regularization prevents v6-style over-opening.
#
#   Dataset fix — balanced perturbation pool (v11):
#     Old: homophones 3 conditions (20/40/50%) = 1500 samples vs 500 for typos/OCR.
#     New: homophones collapsed to 1 condition (30%), OCR gets a second condition
#          at 15% (historically hardest failure type). Total: 7 conditions, balanced.
#
# KEY DIAGNOSTICS: Check at step 3000:
#   Target: emb_mse in range 0.001–0.010 and STABLE or slowly decreasing
#   Target: s1_mx > 0.10 (gate actually open on noisy steps)
#   Target: r1_max > 1.5  (gate selectively higher on noisy vs clean)
#   Target: gnorm > 0.001 consistently
#   Failure: s1_mx still 0.0000 → check gate_bias in src/lib/stabilizer_system.py (-1.0?)
#   Failure: gate opens then collapses after step 3000 → reduce final lambda_suppress
#
# All other hyperparameters unchanged from v10.
python src/lib/train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  mixed \
    --output_dir ./stabilizer_weights/phi35_mixed_mlp_v11 \
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
    --lambda_gate_start  0.02 \
    --lambda_suppress    0.35 \
    --lambda_suppress_start 0.02 \
    --lambda_adapter     0.15 \
    --probe_layers 9 \
    --ema_decay 0.99 \
    --reg_warmup_steps    2000 \
    --reg_decay_end_steps 8000 \
    --reg_ramp_end_steps  3000 \
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
