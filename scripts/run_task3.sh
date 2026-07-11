#!/usr/bin/env bash
# run_task3.sh — Task 3: MMLU layer sweep.
#
# Usage (start ASAP — longest wall-clock; each in its own tmux window):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_task3.sh phi35 0  # Phi-3.5 windows 0-2 × 3 seeds
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_task3.sh phi35 1  # Phi-3.5 windows 3-5 × 3 seeds
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_task3.sh qwen  0  # Qwen windows 0-2 × 3 seeds
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_task3.sh qwen  1  # Qwen windows 3-5 × 3 seeds
#
# Each invocation runs half the windows (3 windows × 3 seeds = 9 runs) on one GPU.
# Results: results/fixed_harness/v2/mmlu/<model>_<WIN>_seed<S>.json
# Checkpoints: /data/nathanlabiosa/nlp2026_weights/phase2_mmlu_<model>_<WIN>_seed<S>/

set -euo pipefail

MODEL_ARG="${1:-phi35}"
HALF="${2:-0}"   # 0 = first 3 windows, 1 = last 3 windows
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_ROOT="/data/nathanlabiosa/nlp2026_weights"
RESULTS_DIR="${ROOT}/results/fixed_harness/v2/mmlu"
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
# Set HF_TOKEN for gated models. Phi-3.5 + Qwen are ungated.
# export HF_TOKEN="<paste_valid_token_here>"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ROOT}/evaluation:${ROOT}/src/lib:$PYTHONPATH"

SEEDS=(42 43 44)
ITEM_IDS_FILE="${ROOT}/results/mmlu_eval_item_ids.json"

# ── Model config ───────────────────────────────────────────────────────────────
case "$MODEL_ARG" in
    phi35)
        MODEL_ID="microsoft/Phi-3.5-mini-instruct"
        MODEL_SLUG="phi35"
        ATTN_IMPL="sdpa"  # eager breaks with transformers>=4.46 (mask mismatch in modeling_phi3.py)
        TARGET_MODULES=("qkv_proj" "o_proj")
        # 6 windows from Table 4
        ALL_STARTS=(0  5  10 15 20 27)
        ALL_ENDS=(  4  9  14 19 24 31)
        ALL_TAGS=("L00-04" "L05-09" "L10-14" "L15-19" "L20-24" "L27-31")
        ;;
    qwen)
        MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
        MODEL_SLUG="qwen2.5_7b"
        ATTN_IMPL="sdpa"
        TARGET_MODULES=("q_proj" "v_proj")
        # 6 windows from Table 4
        ALL_STARTS=(0  5  8  15 20 24)
        ALL_ENDS=(  4  9  11 19 23 27)
        ALL_TAGS=("L00-04" "L05-09" "L08-11" "L15-19" "L20-23" "L24-27")
        ;;
    *)
        echo "Unknown model: $MODEL_ARG. Use phi35 | qwen"
        exit 1
        ;;
esac

# Select first or second half of windows
if [ "$HALF" -eq 0 ]; then
    WIN_INDICES=(0 1 2)
else
    WIN_INDICES=(3 4 5)
fi

echo "============================================"
echo "Task 3 — MMLU sweep: $MODEL_ARG half=$HALF"
echo "  Model:   $MODEL_ID"
echo "  Windows: ${WIN_INDICES[*]} of 6"
echo "  Seeds:   ${SEEDS[*]}"
echo "  GPU:     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================"

for W_IDX in "${WIN_INDICES[@]}"; do
    WIN_START="${ALL_STARTS[$W_IDX]}"
    WIN_END="${ALL_ENDS[$W_IDX]}"
    WIN_TAG="${ALL_TAGS[$W_IDX]}"
    WIDTH=$(( WIN_END - WIN_START + 1 ))

    for SEED in "${SEEDS[@]}"; do
        RUN_TAG="phase2_mmlu_${MODEL_SLUG}_${WIN_TAG}_seed${SEED}"
        TRAIN_DIR="${WEIGHTS_ROOT}/${RUN_TAG}"
        EVAL_OUT="${RESULTS_DIR}/${MODEL_SLUG}_${WIN_TAG}_seed${SEED}.json"
        LOG_FILE="${LOGS_DIR}/${RUN_TAG}.log"

        echo ""
        echo "--- $WIN_TAG seed=$SEED ---"
        echo "  Train dir: $TRAIN_DIR"
        echo "  Eval out:  $EVAL_OUT"

        if [ -f "$EVAL_OUT" ]; then
            DELTA=$(python3 -c "import json; d=json.load(open('$EVAL_OUT')); print(d.get('mean_perturbed_delta','?'))" 2>/dev/null || echo "?")
            echo "  [SKIP] Already done (mean_perturbed_delta=$DELTA)"
            continue
        fi

        mkdir -p "$TRAIN_DIR"

        # ── 1. Train on perturbed MMLU (clean supervision) ─────────────────
        echo "  [1/2] Training on MMLU (dataset=mmlu)..."
        "$PY" "${ROOT}/src/training/train_lrd_lora_v14.py" \
            --model              "$MODEL_ID" \
            --dataset            mmlu \
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
            --clean_eval_every   99999 \
            --clean_eval_n       50 \
            --early_stop_delta   0.0 \
            --no_eval_after_training \
            --seed               $SEED \
            --output_dir         "$TRAIN_DIR" \
            2>&1 | tee "${LOG_FILE}.train"

        if [ ! -d "${TRAIN_DIR}/lora_final" ]; then
            echo "  [ERROR] Training did not produce lora_final — skipping eval"
            continue
        fi

        # ── 2. Eval on fixed MMLU subset (max_new_tokens=32) ──────────────
        echo "  [2/2] Evaluating on MMLU (max_new_tokens=32)..."
        "$PY" "${ROOT}/src/eval/eval_fixed_harness_mmlu.py" \
            --base_model     "$MODEL_ID" \
            --checkpoint     "${TRAIN_DIR}/lora_final" \
            --output_file    "$EVAL_OUT" \
            --item_ids_file  "$ITEM_IDS_FILE" \
            --n_items        500 \
            --batch_size     16 \
            --max_new_tokens 32 \
            --seed           42 \
            --attn_impl      "$ATTN_IMPL" \
            2>&1 | tee "${LOG_FILE}.eval"

        echo "  [DONE] $EVAL_OUT"
    done
done

echo ""
echo "============================================"
echo "Task 3 $MODEL_ARG half=$HALF complete."
echo "Run: python scripts/aggregate_results.py task3"
echo "============================================"
