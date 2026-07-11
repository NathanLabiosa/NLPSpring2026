#!/usr/bin/env bash
# run_task1.sh — Task 1: Window-length ablation on Phi-3.5-mini.
#
# Usage (each in its own tmux window):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_task1.sh w3_mid   # width-3, mid window L11-13
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_task1.sh w3_late  # width-3, late window L28-30
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_task1.sh w7_mid   # width-7, mid window L09-15
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_task1.sh w7_late  # width-7, late window L25-31
#
# Each invocation runs 3 seeds (42,43,44) sequentially on the assigned GPU.
# Results: results/fixed_harness/v2/layer_sweep/phi35_<WIN>_seed<S>.json
# Checkpoints: /data/nathanlabiosa/nlp2026_weights/phase2_phi35_<WIN>_seed<S>/

set -euo pipefail

VARIANT="${1:-w3_mid}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_ROOT="/data/nathanlabiosa/nlp2026_weights"
RESULTS_DIR="${ROOT}/results/fixed_harness/v2/layer_sweep"
LOGS_DIR="${ROOT}/logs"
mkdir -p "$WEIGHTS_ROOT" "$RESULTS_DIR" "$LOGS_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
NLP2026_PY="/home/nathanlabiosa/miniforge3/envs/nlp2026/bin/python"
FALLBACK_PY="/home/nathanlabiosa/miniforge3/envs/lerobot_smol/bin/python"
# Check if torch is importable in nlp2026 (env may exist but install may be in progress)
if "$NLP2026_PY" -c "import torch" 2>/dev/null; then
    PY="$NLP2026_PY"
else
    PY="$FALLBACK_PY"
    echo "[INFO] nlp2026 env not ready yet, using lerobot_smol"
fi
export HF_HOME="/data/nathanlabiosa/hf_cache"
# Set HF_TOKEN for gated models (Llama-3, Mistral, Gemma). Phi-3.5 + Qwen are ungated.
# export HF_TOKEN="<paste_valid_token_here>"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Make perturbations.py importable
export PYTHONPATH="${ROOT}/evaluation:${ROOT}/src/lib:$PYTHONPATH"

MODEL_ID="microsoft/Phi-3.5-mini-instruct"
MODEL_SLUG="phi35"
ATTN_IMPL="sdpa"  # eager breaks with transformers>=4.46 (mask mismatch in modeling_phi3.py)
TARGET_MODULES=("qkv_proj" "o_proj")
SEEDS=(42 43 44)

# ── Window config by variant ───────────────────────────────────────────────────
case "$VARIANT" in
    w3_mid)
        WIN_START=11; WIN_END=13; WIN_TAG="L11-13"
        NOTE="width-3 mid (center 12)"
        ;;
    w3_late)
        WIN_START=28; WIN_END=30; WIN_TAG="L28-30"
        NOTE="width-3 late (center 29)"
        ;;
    w7_mid)
        WIN_START=9; WIN_END=15; WIN_TAG="L09-15"
        NOTE="width-7 mid (center 12)"
        ;;
    w7_late)
        # Clamped from L26-32: Phi-3.5 has 32 layers (0-31), so L26-32 → L25-31
        WIN_START=25; WIN_END=31; WIN_TAG="L25-31"
        NOTE="width-7 late (center 29, clamped L26-32 → L25-31)"
        ;;
    *)
        echo "Unknown variant: $VARIANT. Use w3_mid | w3_late | w7_mid | w7_late"
        exit 1
        ;;
esac

WIDTH=$(( WIN_END - WIN_START + 1 ))

echo "============================================"
echo "Task 1 — Window-length ablation: $VARIANT"
echo "  Window: $WIN_TAG ($NOTE)"
echo "  Width:  $WIDTH layers"
echo "  Seeds:  ${SEEDS[*]}"
echo "  GPU:    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================"

for SEED in "${SEEDS[@]}"; do
    RUN_TAG="phase2_${MODEL_SLUG}_${WIN_TAG}_seed${SEED}"
    TRAIN_DIR="${WEIGHTS_ROOT}/${RUN_TAG}"
    EVAL_OUT="${RESULTS_DIR}/${MODEL_SLUG}_${WIN_TAG}_seed${SEED}.json"
    LOG_FILE="${LOGS_DIR}/${RUN_TAG}.log"

    echo ""
    echo "--- Seed ${SEED}: ${WIN_TAG} ---"
    echo "  Train dir: $TRAIN_DIR"
    echo "  Eval out:  $EVAL_OUT"
    echo "  Log:       $LOG_FILE"

    if [ -f "$EVAL_OUT" ]; then
        DELTA=$(python3 -c "import json; d=json.load(open('$EVAL_OUT')); print(d.get('mean_perturbed_delta','?'))" 2>/dev/null || echo "?")
        echo "  [SKIP] Already done (mean_perturbed_delta=$DELTA)"
        continue
    fi

    mkdir -p "$TRAIN_DIR"

    # ── 1. Train ─────────────────────────────────────────────────────────────
    if [ -d "${TRAIN_DIR}/lora_final" ]; then
        echo "  [SKIP-TRAIN] lora_final already exists, proceeding to eval"
    else
    echo "  [1/2] Training..."
    "$PY" "${ROOT}/src/training/train_lrd_lora_v14.py" \
        --model              "$MODEL_ID" \
        --dataset            gsm8k \
        --lora_rank          4 \
        --lora_alpha         8 \
        --lora_dropout       0.05 \
        --lora_layers        $WIDTH \
        --lora_layers_start  $WIN_START \
        --target_modules     "${TARGET_MODULES[@]}" \
        --lambda_acc         1.0 \
        --lambda_stab        0.0 \
        --stab_layer         9 \
        --n_per_condition    857 \
        --clean_fraction     0.25 \
        --epochs             30 \
        --max_steps          300 \
        --batch_size         4 \
        --grad_accum_steps   4 \
        --lr                 5e-5 \
        --warmup_ratio       0.05 \
        --max_seq_len        512 \
        --log_every          50 \
        --save_every         300 \
        --eval_every         0 \
        --clean_eval_every   300 \
        --clean_eval_n       200 \
        --early_stop_delta   5.0 \
        --no_eval_after_training \
        --seed               $SEED \
        --output_dir         "$TRAIN_DIR" \
        2>&1 | tee "${LOG_FILE}.train"
    fi  # end skip-train block

    if [ ! -d "${TRAIN_DIR}/lora_final" ]; then
        echo "  [ERROR] Training did not produce lora_final — skipping eval"
        continue
    fi

    # ── 2. Eval (fixed harness, max_new_tokens=768) ──────────────────────────
    echo "  [2/2] Evaluating (max_new_tokens=768)..."
    "$PY" "${ROOT}/src/eval/eval_fixed_harness.py" \
        --base_model     "$MODEL_ID" \
        --checkpoint     "${TRAIN_DIR}/lora_final" \
        --n_samples      500 \
        --seed           42 \
        --attn_impl      "$ATTN_IMPL" \
        --max_new_tokens 768 \
        --output_file    "$EVAL_OUT" \
        2>&1 | tee "${LOG_FILE}.eval"

    echo "  [DONE] $EVAL_OUT"
done

echo ""
echo "============================================"
echo "Task 1 $VARIANT complete."
echo "Run: python scripts/aggregate_results.py task1"
echo "============================================"
