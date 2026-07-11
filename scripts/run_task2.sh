#!/usr/bin/env bash
# run_task2.sh — Task 2: All-layer LoRA baseline.
#
# Usage (each in its own tmux window):
#   CUDA_VISIBLE_DEVICES=4 bash scripts/run_task2.sh phi35   # Phi-3.5 (ungated)
#   CUDA_VISIBLE_DEVICES=5 bash scripts/run_task2.sh qwen    # Qwen2.5-7B (ungated)
#   CUDA_VISIBLE_DEVICES=6 bash scripts/run_task2.sh llama   # Llama-3-8B (needs HF token)
#
# Runs 3 seeds (42, 43, 44) sequentially.
# Results: results/fixed_harness/v2/all_layer/<model>_all_seed<S>.json
# Checkpoints: /data/nathanlabiosa/nlp2026_weights/phase2_alllayer_<model>_seed<S>/

set -euo pipefail

MODEL_ARG="${1:-phi35}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_ROOT="/data/nathanlabiosa/nlp2026_weights"
RESULTS_DIR="${ROOT}/results/fixed_harness/v2/all_layer"
LOGS_DIR="${ROOT}/logs"
mkdir -p "$WEIGHTS_ROOT" "$RESULTS_DIR" "$LOGS_DIR"

NLP2026_PY="/home/nathanlabiosa/miniforge3/envs/nlp2026/bin/python"
FALLBACK_PY="/home/nathanlabiosa/miniforge3/envs/lerobot_smol/bin/python"
if "$NLP2026_PY" -c "import torch" 2>/dev/null; then
    PY="$NLP2026_PY"
else
    PY="$FALLBACK_PY"
    echo "[INFO] nlp2026 env not ready yet, using lerobot_smol"
fi
export HF_HOME="/data/nathanlabiosa/hf_cache"
# Set HF_TOKEN for gated models (Llama-3). Phi-3.5 + Qwen are ungated.
# export HF_TOKEN="<paste_valid_token_here>"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ROOT}/evaluation:${ROOT}/src/lib:$PYTHONPATH"

SEEDS=(42 43 44)

# ── Model config ───────────────────────────────────────────────────────────────
case "$MODEL_ARG" in
    phi35)
        MODEL_ID="microsoft/Phi-3.5-mini-instruct"
        MODEL_SLUG="phi35"
        ATTN_IMPL="sdpa"  # eager breaks with transformers>=4.46
        TARGET_MODULES=("qkv_proj" "o_proj")
        N_LAYERS=32  # Phi-3.5-mini has 32 decoder layers (0-31)
        ;;
    qwen)
        MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
        MODEL_SLUG="qwen2.5_7b"
        ATTN_IMPL="sdpa"
        TARGET_MODULES=("q_proj" "v_proj")
        N_LAYERS=28  # Qwen2.5-7B has 28 decoder layers (0-27)
        ;;
    llama)
        MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
        MODEL_SLUG="llama3_8b"
        ATTN_IMPL="sdpa"
        TARGET_MODULES=("q_proj" "v_proj")
        N_LAYERS=32  # Llama-3-8B has 32 decoder layers (0-31)
        ;;
    *)
        echo "Unknown model: $MODEL_ARG. Use phi35 | qwen | llama"
        exit 1
        ;;
esac

echo "============================================"
echo "Task 2 — All-layer LoRA: $MODEL_ARG"
echo "  Model:  $MODEL_ID"
echo "  Layers: 0 – $(( N_LAYERS - 1 )) ($N_LAYERS total)"
echo "  Seeds:  ${SEEDS[*]}"
echo "  GPU:    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================"

# Build explicit layer index list: 0 1 2 ... N_LAYERS-1
LAYER_INDICES=()
for i in $(seq 0 $(( N_LAYERS - 1 ))); do
    LAYER_INDICES+=($i)
done

for SEED in "${SEEDS[@]}"; do
    RUN_TAG="phase2_alllayer_${MODEL_SLUG}_seed${SEED}"
    TRAIN_DIR="${WEIGHTS_ROOT}/${RUN_TAG}"
    EVAL_OUT="${RESULTS_DIR}/${MODEL_SLUG}_all_seed${SEED}.json"
    LOG_FILE="${LOGS_DIR}/${RUN_TAG}.log"

    echo ""
    echo "--- Seed ${SEED}: all-layer $MODEL_SLUG ---"
    echo "  Train dir: $TRAIN_DIR"
    echo "  Eval out:  $EVAL_OUT"

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
    echo "  [1/2] Training (all $N_LAYERS layers, r=4, α=8)..."
    "$PY" "${ROOT}/src/training/train_lrd_lora_v14.py" \
        --model              "$MODEL_ID" \
        --dataset            gsm8k \
        --lora_rank          4 \
        --lora_alpha         8 \
        --lora_dropout       0.05 \
        --lora_layer_indices "${LAYER_INDICES[@]}" \
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

    # ── 2. Eval ────────────────────────────────────────────────────────────
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
echo "Task 2 $MODEL_ARG complete."
echo "Run: python scripts/aggregate_results.py task2"
echo "============================================"
