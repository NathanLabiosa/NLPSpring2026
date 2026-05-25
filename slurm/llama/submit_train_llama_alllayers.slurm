#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=16:00:00
#SBATCH --output=logs/train_llama_alllayers_%j.out
#SBATCH --error=logs/train_llama_alllayers_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f $HF_HOME/token ] && export HF_TOKEN=$(cat $HF_HOME/token)

echo "--- Llama-3 8B LoRA: all-layer spread-out baseline ---"
nvidia-smi
mkdir -p logs

# Llama concentrated-vs-spread comparison:
# Same rank/alpha/steps as llama_lora_v4 (concentrated, layers 0–8),
# but LoRA spread across all 32 layers. If concentrated >> spread on Llama
# too, the Phi-3.5 finding replicates on a second architecture.
#
# L_stab at layer 9 is within the LoRA range here (all layers), so it
# won't provide meaningful regularization. Disable it for clean comparison
# with the concentrated run (which also showed L_stab ≈ 0 in practice).

python src/training/train_lrd_lora_v14.py \
    --model    meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/llama_alllayers \
    --lora_rank    4 \
    --lora_alpha   8 \
    --lora_dropout 0.05 \
    --lora_layers  32 \
    --lora_layers_start 0 \
    --target_modules q_proj v_proj \
    --lambda_acc   1.0 \
    --lambda_stab  0.0 \
    --stab_layer   12 \
    --n_per_condition 857 \
    --clean_fraction  0.25 \
    --epochs 30 \
    --max_steps 300 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 2e-5 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 25 \
    --save_every 300 \
    --eval_every 0 \
    --clean_eval_every 300 \
    --early_stop_delta 15.0 \
    --clean_eval_n 200 \
    --eval_after_training \
    --eval_n_samples 500
