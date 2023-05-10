#!/bin/bash
#SBATCH --job-name=jets
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=20

source activate pyg
python3 train_euclidean_cpu.py

