#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_bootstrap_n1000_%j.out
#SBATCH --error=logs/eval_bootstrap_n1000_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

echo "--- Bootstrap CIs for Phi-3.5 v17 at n=1000 (tighter CIs) ---"
nvidia-smi
mkdir -p logs

# Experiment 3: Larger eval n for tighter CIs.
# n=500 gives ~±3pp CIs; n=1000 shrinks by ~30% to ~±2pp.
# Whitespace (p=0.098) and case (p=0.387) at n=500 are the weakest conditions;
# n=1000 may push whitespace below 0.05 or confirm it's genuinely marginal.
# No new training needed — reuses existing v17/lora_final checkpoint.

python src/eval/eval_bootstrap.py \
    --checkpoint_dir ./stabilizer_weights/lora_v17/lora_final \
    --n_samples   1000 \
    --n_bootstrap 10000 \
    --output      ./stabilizer_weights/lora_v17/bootstrap_results_n1000.json
