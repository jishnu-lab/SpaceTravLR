#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --job-name=SpaceTravLR
#SBATCH --output=train.txt
#SBATCH --error=train.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-4:00:00

source activate bee
python train.py
