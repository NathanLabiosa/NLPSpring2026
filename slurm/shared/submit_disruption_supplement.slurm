#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=04:00:00
#SBATCH --array=1-2
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/disruption_supp_%a_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/disruption_supp_%a_%j.err

# Supplement — cascade-disruption for the 5-model panel.
# Phi-3.5, Llama-3, Mistral already done (job 8947120 + earlier).
# task 1 = Qwen2.5-7B, task 2 = Gemma-2-9B.

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH

export HF_HOME=/scratch1/labiosa/huggingface_cache
export HF_HUB_CACHE=$HF_HOME/hub
if [ -f $HF_HOME/token_gemma ]; then
    export HF_TOKEN_GEMMA=$(cat $HF_HOME/token_gemma)
fi
if [ -f $HF_HOME/token ]; then
    export HF_TOKEN=$(cat $HF_HOME/token)
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=120

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    MODEL_KEY="qwen"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    MODEL_KEY="gemma"
    # Gemma is gated under the Gemma-specific HF account.
    if [ -n "$HF_TOKEN_GEMMA" ]; then export HF_TOKEN="$HF_TOKEN_GEMMA"; fi
else
    echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
cd $ROOT

echo "=========================================="
echo "Cascade-disruption supplement  model=${MODEL_KEY}"
echo "=========================================="
nvidia-smi

$PY src/experiments/expF_clean_disruption.py --model $MODEL_KEY --n_samples 200 --batch_size 4 --seed 42

echo "--- done: ${MODEL_KEY} ---"
