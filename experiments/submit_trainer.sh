#!/bin/bash
#SBATCH -p gtx1080
#SBATCH --gres=gpu:4
#SBATCH --job-name=multi_jets
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gres-flags=enforce-binding


source activate gnn
python3 jet_tagging.py "configs/jets_config_euc.yaml"
python3 jet_tagging.py "configs/jets_config_lor.yaml"
python3 jet_tagging.py "configs/jets_config_poi.yaml"
