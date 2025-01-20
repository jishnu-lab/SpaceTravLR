#!/bin/bash
#SBATCH -J compare0
#SBATCH --output=logs/compare-0.txt
#SBATCH -t 16:00:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --account=djishnu

source activate bee

echo ${SLURM_JOB_NAME} allocated to ${SLURM_NODELIST}
echo environment $CONDA_DEFAULT_ENV
which python

python auto_co.py