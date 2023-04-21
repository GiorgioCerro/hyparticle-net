#!/bin/bash
#SBATCH --job-name=jets
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4

source activate pyg
python3 train_jets.py
#python3 tuning.py
source deactivate

