#!/bin/bash
#SBATCH -p gtx1080
#SBATCH --gres=gpu:1
#SBATCH --job-name=upfd
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20

source /home/mjad1g20/.bashrc
source activate graph 

#python gpu_train.py
#python train_synthetic.py
python train_UPFD.py

