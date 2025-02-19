#!/bin/bash
#SBATCH -J perturb
#SBATCH --output=perturb.txt
#SBATCH -t 4:00:00
#SBATCH --mem=250G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --account=djishnu

# source activate seacells
source activate bee
# source activate spectra

echo ${SLURM_JOB_NAME} allocated to ${SLURM_NODELIST}
echo environment $CONDA_DEFAULT_ENV
which python

# python compare_co.py Foxp3
python perturb.py
# python spectra.py 