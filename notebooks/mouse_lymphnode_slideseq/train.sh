#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --cluster=htc
#SBATCH --mem=250G
#SBATCH --job-name=ScreenLymph
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00

mamba activate SpaceOracle
python train.py
