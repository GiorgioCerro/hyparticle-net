#!/bin/bash
#SBATCH -p gtx1080
#SBATCH --gres=gpu:1
#SBATCH --job-name=jetsgpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20

source activate pyg
python3 train_jets.py
