#!/bin/bash
#SBATCH --cluster=htc
##SBATCH --partition=preempt
#SBATCH --job-name=perturb
#SBATCH --output=perturb.txt
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH --cluster=gpu
##SBATCH --gres=gpu:1
#SBATCH --time=0-8:00:00


conda init
source ~/.bashrc
conda activate /ix3/djishnu/alw399/envs/sheep
echo $CONDA_DEFAULT_ENV

# python train_COVET.py
python perturb.py