#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=SpaceTravLR
#SBATCH --output=train_COVET.txt
##SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-8:00:00

# mamba activate SpaceOracle
conda init
source ~/.bashrc
conda activate bee
echo $CONDA_DEFAULT_ENV
python train_COVET.py
