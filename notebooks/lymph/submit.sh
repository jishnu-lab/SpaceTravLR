#!/bin/bash
#SBATCH -J evaluate
#SBATCH --output=evaluate.txt
#SBATCH -t 3-16:00:00
#SBATCH --mem=230G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --account=djishnu

source activate seacells

echo ${SLURM_JOB_NAME} allocated to ${SLURM_NODELIST}
echo environment $CONDA_DEFAULT_ENV
which python

python perturb.py