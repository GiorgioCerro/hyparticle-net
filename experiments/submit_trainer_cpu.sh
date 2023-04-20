#!/bin/bash
#SBATCH --job-name=jets
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10

source activate pyg
python3 train_jets.py
#python3 tuning.py
source deactivate

