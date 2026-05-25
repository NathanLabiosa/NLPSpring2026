#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=16:00:00
#SBATCH --output=logs/train_llama_lora_ablate_%j.out
#SBATCH --error=logs/train_llama_lora_ablate_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f $HF_HOME/token ] && export HF_TOKEN=$(cat $HF_HOME/token)

echo "--- Llama-3 8B LoRA ablation: uninformed placement (layers 24–31, late layers) ---"
nvidia-smi
mkdir -p logs

# Experiment 2: LoRA at late layers (24–31), OUTSIDE the confirmed causal window.
# Validates that diagnostic-informed layer placement outperforms uninformed placement.
# NOTE: L_stab gets no gradient signal here because stab_layer=9 is before the LoRA
# layers (24–31) — the hook captures identical states in frozen and adapted paths.
# Training effectively uses L_acc only (same as if --lambda_stab 0), which is the
# fair comparison: can late-layer LoRA recover task accuracy without access to the
# causal window?
#
# Run AFTER Experiment 1 (llama_lora_v1) succeeds. A weaker result here validates
# the causal window finding on a second architecture.

python src/training/train_lrd_lora_v14.py \
    --model    meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/llama_lora_ablate \
    --lora_rank    4 \
    --lora_alpha   8 \
    --lora_dropout 0.05 \
    --lora_layers  8 \
    --lora_layers_start 24 \
    --target_modules q_proj v_proj \
    --lambda_acc   0.7 \
    --lambda_stab  0.3 \
    --stab_layer   9 \
    --n_per_condition 857 \
    --clean_fraction  0.25 \
    --epochs 30 \
    --max_steps 300 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 5e-5 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 25 \
    --save_every 100 \
    --eval_every 0 \
    --clean_eval_every 100 \
    --early_stop_delta 5.0 \
    --clean_eval_n 200 \
    --eval_after_training \
    --eval_n_samples 500
