#!/usr/bin/env bash
# run_scatter_train.sh — Train + eval for Llama-3 and Mistral scatter checkpoints.
#
# Phi-3.5 and Qwen already have checkpoints — use run_scatter_eval.sh directly.
# This script trains 3 seeds for a given model + window, then runs eval.
#
# Llama-3 is a gated model — export HF_TOKEN in your shell before running.
#
# Usage (one tmux window per window, 6 per model):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_scatter_train.sh llama   L00-04
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_scatter_train.sh llama   L05-09
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_scatter_train.sh llama   L10-14
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_scatter_train.sh llama   L15-19
#   CUDA_VISIBLE_DEVICES=4 bash scripts/run_scatter_train.sh llama   L20-24
#   CUDA_VISIBLE_DEVICES=5 bash scripts/run_scatter_train.sh llama   L27-31
#   (same pattern for mistral)
#
# Checkpoints: /data/nathanlabiosa/nlp2026_weights/phase2_scatter_{model}_{window}_seed{S}/
# Results:     results/fixed_harness/v2/scatter/{slug}_{window}_seed{S}.json

set -euo pipefail

MODEL_ARG="${1:?Usage: run_scatter_train.sh MODEL WINDOW}"
WINDOW="${2:?Usage: run_scatter_train.sh MODEL WINDOW}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_ROOT="/data/nathanlabiosa/nlp2026_weights"
RESULTS_DIR="${ROOT}/results/fixed_harness/v2/scatter"
LOGS_DIR="${ROOT}/logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

NLP2026_PY="/home/nathanlabiosa/miniforge3/envs/nlp2026/bin/python"
FALLBACK_PY="/home/nathanlabiosa/miniforge3/envs/lerobot_smol/bin/python"
if "$NLP2026_PY" -c "import torch" 2>/dev/null; then
    PY="$NLP2026_PY"
else
    PY="$FALLBACK_PY"
    echo "[INFO] nlp2026 env not ready, using lerobot_smol"
fi

export HF_HOME="/data/nathanlabiosa/hf_cache"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ROOT}/evaluation:${ROOT}/src/lib:$PYTHONPATH"

SEEDS=(42 43 44)

case "$MODEL_ARG" in
    phi35)
        MODEL_ID="microsoft/Phi-3.5-mini-instruct"
        MODEL_SLUG="phi35"
        ATTN_IMPL="sdpa"
        TARGET_MODULES=("qkv_proj" "o_proj")
        N_LAYERS=32
        CKPT_PREFIX="phase2_scatter_phi35"
        ;;
    qwen)
        MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
        MODEL_SLUG="qwen2.5_7b"
        ATTN_IMPL="sdpa"
        TARGET_MODULES=("q_proj" "v_proj")
        N_LAYERS=28
        CKPT_PREFIX="phase2_scatter_qwen2.5_7b"
        ;;
    llama)
        MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
        MODEL_SLUG="llama3_8b"
        ATTN_IMPL="sdpa"
        TARGET_MODULES=("q_proj" "v_proj")
        N_LAYERS=32
        CKPT_PREFIX="phase2_scatter_llama3_8b"
        ;;
    mistral)
        MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
        MODEL_SLUG="mistral_7b_v03"
        ATTN_IMPL="sdpa"
        TARGET_MODULES=("q_proj" "v_proj")
        N_LAYERS=32
        CKPT_PREFIX="phase2_scatter_mistral_7b_v03"
        ;;
    *)
        echo "Unknown model: $MODEL_ARG. Use phi35 | qwen | llama | mistral"
        exit 1
        ;;
esac

# Parse window start/end/width from tag like L00-04
WIN_START_RAW=$(echo "$WINDOW" | sed 's/L\([0-9]*\)-[0-9]*/\1/')
WIN_END_RAW=$(echo "$WINDOW"   | sed 's/L[0-9]*-\([0-9]*\)/\1/')
# Strip leading zeros (bash arithmetic doesn't support octal here)
WIN_START=$(echo "$WIN_START_RAW" | sed 's/^0*//' ); WIN_START=${WIN_START:-0}
WIN_END=$(echo "$WIN_END_RAW"     | sed 's/^0*//' ); WIN_END=${WIN_END:-0}
WIDTH=$(( WIN_END - WIN_START + 1 ))

echo "============================================"
echo "Scatter train+eval: $MODEL_ARG $WINDOW"
echo "  Layers: ${WIN_START}-${WIN_END} (width ${WIDTH})"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================"

for SEED in "${SEEDS[@]}"; do
    RUN_TAG="${CKPT_PREFIX}_${WINDOW}_seed${SEED}"
    TRAIN_DIR="${WEIGHTS_ROOT}/${RUN_TAG}"
    EVAL_OUT="${RESULTS_DIR}/${MODEL_SLUG}_${WINDOW}_seed${SEED}.json"
    LOG_BASE="${LOGS_DIR}/${RUN_TAG}"

    echo ""
    echo "--- Seed ${SEED}: ${WINDOW} ---"

    if [ -f "$EVAL_OUT" ]; then
        DELTA=$("$PY" -c "import json; d=json.load(open('$EVAL_OUT')); print(d.get('mean_perturbed_delta','?'))" 2>/dev/null || echo "?")
        echo "  [SKIP] eval already done (delta=$DELTA)"
        continue
    fi

    mkdir -p "$TRAIN_DIR"

    # ── 1. Train ─────────────────────────────────────────────────────────────
    if [ -d "${TRAIN_DIR}/lora_final" ]; then
        echo "  [SKIP-TRAIN] lora_final exists, running eval"
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
            2>&1 | tee "${LOG_BASE}.train"
    fi

    if [ ! -d "${TRAIN_DIR}/lora_final" ]; then
        echo "  [ERROR] Training did not produce lora_final — skipping eval"
        continue
    fi

    # ── 2. Eval ───────────────────────────────────────────────────────────────
    echo "  [2/2] Evaluating (max_new_tokens=768)..."
    "$PY" "${ROOT}/src/eval/eval_fixed_harness.py" \
        --base_model     "$MODEL_ID" \
        --checkpoint     "${TRAIN_DIR}/lora_final" \
        --n_samples      500 \
        --seed           42 \
        --attn_impl      "$ATTN_IMPL" \
        --max_new_tokens 768 \
        --output_file    "$EVAL_OUT" \
        2>&1 | tee "${LOG_BASE}.eval"

    echo "  [DONE] $EVAL_OUT"
done

echo ""
echo "============================================"
echo "Scatter $MODEL_ARG $WINDOW complete."
echo "============================================"
