#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --job-name=SpaceTravLR_human_tonsil
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-8:00:00

mamba activate SpaceOracle
python train.py
