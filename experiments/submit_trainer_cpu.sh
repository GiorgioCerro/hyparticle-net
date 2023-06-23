#!/bin/bash
#SBATCH --job-name=jets
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=40

source activate gnn
python3 train_jets.py
