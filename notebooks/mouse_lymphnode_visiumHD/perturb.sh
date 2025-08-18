#!/bin/bash
#SBATCH --job-name=lymph1c_COVET
#SBATCH --output=lymph1c_COVET.out
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G                   
#SBATCH --time=16:00:00               
#SBATCH --partition=htc

conda init
source ~/.bashrc
conda activate bee
echo $CONDA_DEFAULT_ENV

python perturb_COVET.py --sample lymph1c
