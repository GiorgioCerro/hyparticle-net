#!/bin/bash

#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=3                # Number of nodes requested
#SBATCH --time=15:00:00          # walltime

# mail alert at start, end and abortion
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=g.cerro@soton.ac.uk

source activate pyg
mpiexec -n 120 python3 make_dataset.py ../../data/signal/hz_train.lhe.gz pythia-settings.cmnd ../../data/signal/hz_train.hdf5 signal
source deactivate
