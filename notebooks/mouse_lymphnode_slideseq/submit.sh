#!/bin/bash
#SBATCH -J Runx1
#SBATCH --output=compare.txt
#SBATCH -t 6:00:00
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --account=djishnu

source activate bee

echo ${SLURM_JOB_NAME} allocated to ${SLURM_NODELIST}
echo environment $CONDA_DEFAULT_ENV
which python

python compare_co.py
exit()