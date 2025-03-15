#!/bin/bash
#SBATCH --partition=htc
#SBATCH --job-name=GenomeScreen
#SBATCH --output=/dev/null
##SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=200G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-18:00:00

mamba activate SpaceOracle
python train.py
