#!/bin/bash
#SBATCH -p gtx1080
#SBATCH --gres=gpu:1
#SBATCH --job-name=trainer
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20

source /home/mjad1g20/.bashrc
source activate graph 

python gcn_classifier.py


