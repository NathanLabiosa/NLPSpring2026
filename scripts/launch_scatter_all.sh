#!/usr/bin/env bash
# launch_scatter_all.sh — Queue all 18 scatter training windows across 5 free GPUs.
#
# Each GPU gets one tmux session that runs its assigned windows SEQUENTIALLY.
# GPU 1, 2, 5 skipped (currently occupied).
#
# Usage: bash scripts/launch_scatter_all.sh
# Monitor: tmux ls

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

SCRIPT="$ROOT/scripts/run_scatter_train.sh"

launch_gpu() {
    local SESSION="$1"
    local GPU="$2"
    local LOGFILE="$ROOT/logs/${SESSION}.log"
    shift 2
    # Write a small per-gpu runner script
    local RUNNER="$ROOT/logs/${SESSION}_run.sh"
    {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        echo "export CUDA_VISIBLE_DEVICES=$GPU"
        for ARG_PAIR in "$@"; do
            M="${ARG_PAIR%% *}"
            W="${ARG_PAIR##* }"
            echo "echo '--- Starting $M $W on GPU $GPU ---'"
            echo "bash '$SCRIPT' $M $W"
        done
        echo "echo '=== GPU $GPU queue complete ==='"
    } > "$RUNNER"
    chmod +x "$RUNNER"

    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" "bash '$RUNNER' 2>&1 | tee '$LOGFILE'"
    echo "  $SESSION (GPU $GPU): $*"
}

echo "=== Launching scatter training queues ==="
launch_gpu "gpu0_scatter" 0 \
    "phi35 L00-04" "qwen L00-04" "mistral L00-04" "mistral L15-19"

launch_gpu "gpu3_scatter" 3 \
    "phi35 L05-09" "qwen L05-09" "mistral L05-09" "mistral L20-24"

launch_gpu "gpu4_scatter" 4 \
    "phi35 L10-14" "qwen L08-11" "mistral L10-14" "mistral L27-31"

launch_gpu "gpu6_scatter" 6 \
    "phi35 L15-19" "qwen L15-19" "phi35 L27-31"

launch_gpu "gpu7_scatter" 7 \
    "phi35 L20-24" "qwen L20-23" "qwen L24-27"

echo ""
echo "Active sessions:"
tmux ls
echo ""
echo "Monitor: tmux attach -t gpu0_scatter"
echo ""
echo "After all complete, run disruption then merge:"
echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/run_scatter_disruption.sh phi35"
echo "  CUDA_VISIBLE_DEVICES=3 bash scripts/run_scatter_disruption.sh qwen"
echo "  CUDA_VISIBLE_DEVICES=4 bash scripts/run_scatter_disruption.sh llama"
echo "  CUDA_VISIBLE_DEVICES=6 bash scripts/run_scatter_disruption.sh mistral"
echo "  python scripts/update_scatter_deltas.py"
