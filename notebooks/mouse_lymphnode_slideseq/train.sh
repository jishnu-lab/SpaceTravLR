#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=SpaceTravLR
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00

export OMP_NUM_THREADS=4

mamba activate SpaceOracle

for i in {1..$OMP_NUM_THREADS}
do
    python train.py
done
