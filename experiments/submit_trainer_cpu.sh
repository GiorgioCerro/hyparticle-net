#!/bin/bash
#SBATCH --job-name=jets
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=20

source activate pyg
python3 train_jets.py
#python3 tuning.py
source deactivate

