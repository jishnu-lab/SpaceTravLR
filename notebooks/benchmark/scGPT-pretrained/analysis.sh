#!/bin/bash
##SBATCH --partition=preempt
#SBATCH --cluster=htc
#SBATCH --mem=180G
#SBATCH --job-name=scGPT_pretrained
#SBATCH --output=analysis.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

conda init
source ~/.bashrc
conda activate /ix3/djishnu/alw399/envs/sheep
echo $CONDA_DEFAULT_ENV

python analysis.py
