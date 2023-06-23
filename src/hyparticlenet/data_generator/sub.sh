#!/bin/bash

#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=4:00:00           # walltime

pythia_path="pythia-settings.cmnd"

lhe_sig="/scratch/gc2c20/data/jet_tagging/ww.lhe.gz"
output_sig="/scratch/gc2c20/data/jet_tagging/train_sig/sig.hdf5"

lhe_bkg="/scratch/gc2c20/data/jet_tagging/jj.lhe.gz"
output_bkg="/scratch/gc2c20/data/jet_tagging/train_bkg/bkg.hdf5"

source activate gnn
mpiexec -n 40 python3 make_dataset.py $lhe_sig $pythia_path $output_sig signal
mpiexec -n 40 python3 make_dataset.py $lhe_bkg $pythia_path $output_bkg background

lhe_sig="/scratch/gc2c20/data/jet_tagging/ww2.lhe.gz"
output_sig="/scratch/gc2c20/data/jet_tagging/train_sig/sig2.hdf5"
mpiexec -n 40 python3 make_dataset.py $lhe_sig $pythia_path $output_sig signal


