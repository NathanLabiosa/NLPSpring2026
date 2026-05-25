#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=16:00:00
#SBATCH --array=1-3
#SBATCH --output=logs/exp3_lrd_diag_%a_%j.out
#SBATCH --error=logs/exp3_lrd_diag_%a_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
[ -f $HF_HOME/token ] && export HF_TOKEN=$(cat $HF_HOME/token)

mkdir -p logs

# Exp 3: LRD diagnostics at n=500 for all 3 models.
# Produces per-layer LRD curves and final-layer Cohen's d with bootstrap CIs.
# Tightens the effect-size table (currently very wide CIs at n=200).

COMMON_ARGS="--dataset gsm8k \
    --n_samples 500 \
    --max_new_tokens 256 \
    --run_exclusion_analysis \
    --run_taxonomy_variance"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "--- Exp3: LRD diagnostics, Phi-3.5-mini-instruct, n=500 ---"
    cd /home1/labiosa/NLPSpring2026/Phi3.5
    python lrd_diagnostics.py $COMMON_ARGS \
        --model microsoft/Phi-3.5-mini-instruct \
        --output_dir lrd_results/exp3_phi_n500

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "--- Exp3: LRD diagnostics, Llama-3-8B-Instruct, n=500 ---"
    cd /home1/labiosa/NLPSpring2026/Llama
    python lrd_diagnostics.py $COMMON_ARGS \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --output_dir lrd_results/exp3_llama_n500

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    echo "--- Exp3: LRD diagnostics, Mistral-7B-Instruct-v0.3, n=500 ---"
    cd /home1/labiosa/NLPSpring2026/Mistral
    python lrd_diagnostics.py $COMMON_ARGS \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --output_dir lrd_results/exp3_mistral_n500
fi
