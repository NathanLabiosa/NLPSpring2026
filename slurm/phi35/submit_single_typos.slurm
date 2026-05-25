#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/single_typos_%j.out
#SBATCH --error=logs/single_typos_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- SINGLE-TYPE STABILIZER: typos ---"
nvidia-smi
mkdir -p logs

python src/lib/train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  mixed \
    --output_dir ./stabilizer_weights/single_typos \
    --perturbation_types typos \
    --inject_layers 2 4 \
    --stage2_bottleneck_dim 32 \
    --no_gate \
    --no_embed_stage \
    --max_norm 0.025 \
    --lambda_embed_mse   0.0 \
    --lambda_stab        0.5 \
    --lambda_acc         0.4 \
    --lambda_prop        0.1 \
    --lambda_gate        0.0 \
    --lambda_suppress    0.0 \
    --lambda_adapter     0.0 \
    --probe_layers 9 \
    --stab_warmup_frac   0.10 \
    --gate_warmup_steps  0 \
    --epochs 20 \
    --max_steps 5000 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --n_per_condition 500 \
    --max_seq_len 512 \
    --log_every 25 \
    --save_every 500 \
    --eval_after_training \
    --eval_n_samples 100
