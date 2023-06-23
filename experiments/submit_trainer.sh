#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=multi_jets
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gres-flags=enforce-binding

source activate gnn
python3 jet_tagging.py "configs/jets_config.yaml"
#python3 tuning.py
