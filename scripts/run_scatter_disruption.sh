#!/usr/bin/env bash
# run_scatter_disruption.sh — Measure LoRA-induced clean disruption for scatter.
#
# Runs expF_clean_disruption.py on seed-42 checkpoints for all 6 windows of a
# model, then saves per-layer disruption curves to results/expF_clean_disruption_{model}.json.
# Run AFTER the checkpoints exist (after run_scatter_eval.sh / run_scatter_train.sh).
# Run BEFORE update_scatter_deltas.py (which fills in the acc_delta values).
#
# Usage (one GPU per model, can run all 4 in parallel):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_scatter_disruption.sh phi35
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_scatter_disruption.sh qwen
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_scatter_disruption.sh llama
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_scatter_disruption.sh mistral

set -euo pipefail

MODEL_ARG="${1:?Usage: run_scatter_disruption.sh MODEL}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_ROOT="/data/nathanlabiosa/nlp2026_weights"
LOGS_DIR="${ROOT}/logs"
mkdir -p "$LOGS_DIR"

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

case "$MODEL_ARG" in
    phi35|qwen|llama|mistral) ;;
    *) echo "Unknown model: $MODEL_ARG. Use phi35 | qwen | llama | mistral"; exit 1 ;;
esac

# expF_clean_disruption.py's --model choices are phi35/llama3/mistral/qwen/gemma
EXPF_MODEL_ARG="$MODEL_ARG"
if [ "$MODEL_ARG" = "llama" ]; then
    EXPF_MODEL_ARG="llama3"
fi

echo "============================================"
echo "Scatter disruption: $MODEL_ARG"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Weights dir: $WEIGHTS_ROOT"
echo "============================================"

"$PY" "${ROOT}/src/experiments/expF_clean_disruption.py" \
    --model      "$EXPF_MODEL_ARG" \
    --weights_dir "$WEIGHTS_ROOT" \
    --n_samples  200 \
    --batch_size 4 \
    2>&1 | tee "${LOGS_DIR}/scatter_disruption_${MODEL_ARG}.log"

echo ""
echo "Disruption measurement complete: results/expF_clean_disruption_${EXPF_MODEL_ARG}.json"
echo "Next: python scripts/update_scatter_deltas.py"
