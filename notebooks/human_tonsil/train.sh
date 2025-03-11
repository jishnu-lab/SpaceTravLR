#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=SpaceTravLR
#SBATCH --output=train.txt
##SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-2:00:00

# mamba activate SpaceOracle
# conda init 
conda activate bee
python train.py
