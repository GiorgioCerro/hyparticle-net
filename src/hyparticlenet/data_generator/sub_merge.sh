#!/bin/bash

#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=12:00:00           # walltime

source activate gnn
python3 merge_files.py
