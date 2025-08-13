#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --cluster=gpu
#SBATCH --mem=100G
#SBATCH --job-name=lymphCOVET
#SBATCH --output=lymphCOVET.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00


conda init
source ~/.bashrc
conda activate bee
echo $CONDA_DEFAULT_ENV

# python train_KO.py
python train_COVET.py
