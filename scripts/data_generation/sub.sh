#!/bin/bash

#SBATCH --ntasks-per-node=25     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=00:10:00          # walltime

source activate pyg
mpiexec -n 25 python3 make_dataset.py ../data/signal/hz_train.lhe.gz pythia-settings.cmnd ../data/signal/hz_train.hdf5
source deactivate
