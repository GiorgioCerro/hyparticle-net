#!/bin/bash
#SBATCH --job-name=jets
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20

source /home/mjad1g20/.bashrc
source activate graph 

#python train_synthetic.py
#python train_UPFD.py
python train_jets.py

