#!/bin/bash
#SBATCH --partition=htc
#SBATCH --job-name=SpaceTravLR
#SBATCH --output=train.txt
##SBATCH --error=train.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=1
##SBATCH --cluster=gpu
##SBATCH --gres=gpu:1
#SBATCH --time=0-8:00:00

# mamba activate SpaceOracle
# python train.py

conda init
source ~/.bashrc
conda activate bee
echo $CONDA_DEFAULT_ENV
python train_COVET.py