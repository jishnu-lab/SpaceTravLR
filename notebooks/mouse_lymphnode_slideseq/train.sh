#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --cluster=gpu
#SBATCH --mem=50G
#SBATCH --job-name=covet_lymph
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# mamba activate SpaceOracle
# python train.py


conda init
source ~/.bashrc
conda activate bee
echo $CONDA_DEFAULT_ENV

python train_COVET.py
