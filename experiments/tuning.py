import sys

import optuna
from optuna.trial import TrialState

from torch.utils.data import DataLoader
from lundnet.dgl_dataset import DGLGraphDatasetLund, collate_wrapper_tree

from hyparticlenet.hgnn.train import train

import torch

from tqdm import tqdm
import random

from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")


#NUM_THREADS = 4
#torch.set_num_threads = NUM_THREADS

PATH = '/scratch/gc2c20/data/w_tagging/'
train_dataset = DGLGraphDatasetLund(PATH+'/train_bkg/', PATH+'/train_sig/', 
        nev=-1, n_samples=200_000)
valid_dataset = DGLGraphDatasetLund(PATH+'/valid_bkg/', PATH+'/valid_sig/', 
        nev=-1, n_samples=20_000)



def objective(trial):
    config_path = 'configs/jets_config.yaml'
    args = OmegaConf.load(config_path)
    args.manifold = 'lorentz'

    args.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    args.batch_size = trial.suggest_int("batch_size", 8, 128)
    #args.num_layers = trial.suggest_int("num_layers", 3, 6)
    #args.embed_dim = trial.suggest_int("embed_dim", 3, 100)
    args.num_centroid = trial.suggest_int("num_centroid", 10, 250)
    args.optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    args.epochs = trial.suggest_int("epochs", 30, 70)

    collate_fn = collate_wrapper_tree
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn)

    accuracy = train(train_loader, val_loader, args)
    return accuracy

    
if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
