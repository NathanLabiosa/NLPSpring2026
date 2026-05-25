#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_%j.out #
#SBATCH --error=logs/run_%j.err

# Load environment
module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- STARTING PIPELINE ---"
nvidia-smi


#python main.py --dataset squad

#python train_stabilizer.py --inject_layer 16 --epochs 3 --output_dir ./stabilizer_weights --corpus_dir ./stabilizer_corpus

#python lrd_diagnostics.py --dataset bbh --n_samples 400 --patching --n_pairs 40 --output_dir "lrd_bbh"

python patch_only.py
