#!/usr/bin/env bash
# run_scatter_eval.sh — GSM8K eval for scatter figure checkpoints.
#
# Phi-3.5 and Qwen use existing phase2_mmlu checkpoints (no training needed).
# Llama-3 and Mistral use phase2_scatter checkpoints (must be trained first
# via run_scatter_train.sh).
#
# Usage (one tmux window per window, 6 per model):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_scatter_eval.sh phi35   L00-04
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_scatter_eval.sh phi35   L05-09
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_scatter_eval.sh phi35   L10-14
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_scatter_eval.sh phi35   L15-19
#   CUDA_VISIBLE_DEVICES=4 bash scripts/run_scatter_eval.sh phi35   L20-24
#   CUDA_VISIBLE_DEVICES=5 bash scripts/run_scatter_eval.sh phi35   L27-31
#   (same pattern for qwen, llama, mistral)
#
# Results: results/fixed_harness/v2/scatter/{slug}_{window}_seed{S}.json

set -euo pipefail

MODEL_ARG="${1:?Usage: run_scatter_eval.sh MODEL WINDOW}"
WINDOW="${2:?Usage: run_scatter_eval.sh MODEL WINDOW}"

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
        CKPT_PREFIX="phase2_mmlu_phi35"
        ;;
    qwen)
        MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
        MODEL_SLUG="qwen2.5_7b"
        ATTN_IMPL="sdpa"
        CKPT_PREFIX="phase2_mmlu_qwen2.5_7b"
        ;;
    llama)
        MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
        MODEL_SLUG="llama3_8b"
        ATTN_IMPL="sdpa"
        CKPT_PREFIX="phase2_scatter_llama3_8b"
        # export HF_TOKEN="<paste_valid_token_here>"
        ;;
    mistral)
        MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
        MODEL_SLUG="mistral_7b_v03"
        ATTN_IMPL="sdpa"
        CKPT_PREFIX="phase2_scatter_mistral_7b_v03"
        # export HF_TOKEN="<paste_valid_token_here>"
        ;;
    *)
        echo "Unknown model: $MODEL_ARG. Use phi35 | qwen | llama | mistral"
        exit 1
        ;;
esac

echo "============================================"
echo "Scatter eval: $MODEL_ARG $WINDOW"
echo "  Checkpoint prefix: $CKPT_PREFIX"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================"

for SEED in "${SEEDS[@]}"; do
    CKPT_DIR="${WEIGHTS_ROOT}/${CKPT_PREFIX}_${WINDOW}_seed${SEED}/lora_final"
    EVAL_OUT="${RESULTS_DIR}/${MODEL_SLUG}_${WINDOW}_seed${SEED}.json"
    LOG_FILE="${LOGS_DIR}/scatter_eval_${MODEL_SLUG}_${WINDOW}_seed${SEED}.log"

    if [ -f "$EVAL_OUT" ]; then
        DELTA=$("$PY" -c "import json; d=json.load(open('$EVAL_OUT')); print(d.get('mean_perturbed_delta','?'))" 2>/dev/null || echo "?")
        echo "  [SKIP] seed${SEED} already done (delta=$DELTA)"
        continue
    fi

    if [ ! -d "$CKPT_DIR" ]; then
        echo "  [ERROR] seed${SEED}: checkpoint not found at $CKPT_DIR"
        echo "          For llama/mistral run run_scatter_train.sh first."
        continue
    fi

    echo "  [eval] seed${SEED}..."
    "$PY" "${ROOT}/src/eval/eval_fixed_harness.py" \
        --base_model     "$MODEL_ID" \
        --checkpoint     "$CKPT_DIR" \
        --n_samples      500 \
        --seed           42 \
        --attn_impl      "$ATTN_IMPL" \
        --max_new_tokens 768 \
        --output_file    "$EVAL_OUT" \
        2>&1 | tee "$LOG_FILE"

    echo "  [DONE] seed${SEED}: $EVAL_OUT"
done

echo ""
echo "Window $WINDOW complete for $MODEL_ARG."
