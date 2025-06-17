#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=SpaceLymph
#SBATCH --mem=250G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

mamba activate SpaceOracle
python train.py
