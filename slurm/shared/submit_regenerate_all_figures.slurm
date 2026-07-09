#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --output=logs/regenerate_figures_%j.out
#SBATCH --error=logs/regenerate_figures_%j.err
#SBATCH --job-name=regen_figs

# =============================================================================
# Regenerate All Publication Figures
# =============================================================================
# This script regenerates all figures for the paper with updated styling:
#   - Publication-quality fonts (serif, 11-13pt)
#   - Thicker lines (2.0-2.5pt)
#   - Removed top/right spines
#   - Per-figure specific improvements
#
# Main paper figures (8):
#   Fig 1: figures/lrd_heatmap_*.pdf (src/plotting/generate_lrd_heatmaps.py)
#   Fig 2: figures/cohens_d_*.pdf (src/experiments/expA_cohens_d.py)
#   Fig 3: figures/patching_recovery.pdf (workstream_c/generate_figures.py)
#   Fig 4: figures/three_map_*.pdf (src/experiments/expB_three_map.py)
#   Fig 5: figures/disruption_*.pdf (src/experiments/expF_clean_disruption.py)
#   Fig 6: figures/disruption_*_scatter.pdf (src/experiments/expF_clean_disruption.py)
#   Fig 7: figures/cross_task_*.pdf (src/experiments/expD_cross_task.py)
#   Fig 8: figures/layer_sweep.pdf (workstream_c/generate_figures.py)
#
# Supplemental figures (14):
#   - Additional heatmaps, Cohen's d, three-maps, disruption, capacity metrics
# =============================================================================

set -e  # Exit on error

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home1/labiosa/NLPSpring2026

# Create logs and figures directories if needed
mkdir -p logs figures

echo "=============================================="
echo "  Regenerating All Publication Figures"
echo "  Started: $(date)"
echo "=============================================="
nvidia-smi
echo ""

# -----------------------------------------------------------------------------
# PART 1: CPU-only figure generation (fast, no GPU needed)
# -----------------------------------------------------------------------------
echo ""
echo ">>> PART 1: CPU-only figures (from existing data)"
echo "----------------------------------------------"

echo "[1/8] LRD Heatmaps (Gemma2, Qwen2.5)..."
python src/plotting/generate_lrd_heatmaps.py
echo "  Done: figures/lrd_heatmap_gemma.pdf, figures/lrd_heatmap_qwen.pdf"

echo "[2/8] Cohen's d profiles (Phi3.5, Llama3)..."
python src/experiments/expA_cohens_d.py
echo "  Done: figures/cohens_d_phi35.pdf, figures/cohens_d_llama3.pdf"

echo "[3/8] Three-map overlay (Phi3.5, Llama3, Mistral)..."
python src/experiments/expB_three_map.py
echo "  Done: figures/three_map_phi35.pdf, figures/three_map_llama3.pdf, figures/three_map_mistral.pdf"

echo "[4/8] Cross-task LRD stability (Phi3.5, Mistral)..."
python src/experiments/expD_cross_task.py
echo "  Done: figures/cross_task_phi35.pdf, figures/cross_task_mistral.pdf"

echo "[5/8] Workstream C figures (patching recovery, layer sweep, etc.)..."
python src/plotting/generate_workstream_c_figures.py
echo "  Done: figures/patching_recovery.pdf, figures/layer_sweep.pdf, figures/interaction_forest.pdf, etc."

# -----------------------------------------------------------------------------
# PART 2: GPU-required figure generation (loads models)
# -----------------------------------------------------------------------------
echo ""
echo ">>> PART 2: GPU-required figures (loading models)"
echo "----------------------------------------------"

echo "[6/8] Clean disruption - Phi3.5..."
python src/experiments/expF_clean_disruption.py --model phi35 --n_samples 200
echo "  Done: figures/disruption_phi35.pdf, figures/disruption_phi35_scatter.pdf"

echo "[7/8] Clean disruption - Llama3..."
python src/experiments/expF_clean_disruption.py --model llama3 --n_samples 200
echo "  Done: figures/disruption_llama3.pdf, figures/disruption_llama3_scatter.pdf"

echo "[8/8] Clean disruption - Mistral..."
python src/experiments/expF_clean_disruption.py --model mistral --n_samples 200
echo "  Done: figures/disruption_mistral.pdf, figures/disruption_mistral_scatter.pdf"

# -----------------------------------------------------------------------------
# PART 3: Capacity metrics (GPU, longer runtime)
# -----------------------------------------------------------------------------
echo ""
echo ">>> PART 3: Capacity metrics (supplemental)"
echo "----------------------------------------------"

echo "[S1] Capacity metrics - Phi3.5..."
python src/experiments/expC_capacity_metrics.py --model phi35 --n_samples 200
echo "  Done: figures/capacity_metrics_phi35.pdf"

echo "[S2] Capacity metrics - Llama3..."
python src/experiments/expC_capacity_metrics.py --model llama3 --n_samples 200
echo "  Done: figures/capacity_metrics_llama3.pdf"

echo "[S3] Capacity metrics - Mistral..."
python src/experiments/expC_capacity_metrics.py --model mistral --n_samples 200
echo "  Done: figures/capacity_metrics_mistral.pdf"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  All figures regenerated successfully!"
echo "  Finished: $(date)"
echo "=============================================="
echo ""
echo "Main paper figures:"
ls -lh figures/lrd_heatmap_*.pdf 2>/dev/null || echo "  (lrd heatmaps not found)"
ls -lh figures/cohens_d_*.pdf 2>/dev/null || echo "  (cohens_d not found)"
ls -lh figures/patching_recovery.pdf 2>/dev/null || echo "  (patching_recovery not found)"
ls -lh figures/three_map_*.pdf 2>/dev/null || echo "  (three_map not found)"
ls -lh figures/disruption_phi35*.pdf 2>/dev/null || echo "  (disruption_phi35 not found)"
ls -lh figures/cross_task_*.pdf 2>/dev/null || echo "  (cross_task not found)"
ls -lh figures/layer_sweep.pdf 2>/dev/null || echo "  (layer_sweep not found)"
echo ""
echo "Supplemental figures:"
ls -lh figures/disruption_llama3*.pdf figures/disruption_mistral*.pdf 2>/dev/null || true
ls -lh figures/capacity_metrics_*.pdf 2>/dev/null || true
ls -lh figures/interaction_forest.pdf 2>/dev/null || true
echo ""
echo "Total figures in directory:"
ls -1 figures/*.pdf 2>/dev/null | wc -l
