import wandb
from pathlib import Path
from dgl.dataloading import GraphDataLoader
from torch.utils.data import ConcatDataset, Subset

from hyparticlenet.hgnn.train import train
from hyparticlenet.hgnn.util import wandb_cluster_mode, collate_fn, worker_init_fn
from hyparticlenet.data_handler import ParticleDataset

from omegaconf import OmegaConf
config_path = 'configs/jets_config.yaml'
args = OmegaConf.load(config_path)


#import torch
#NUM_THREADS = 4
#torch.set_num_threads = NUM_THREADS

import warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")

# Jets Data sets
PATH = '/scratch/gc2c20/data/jet_tagging/prova'

args.epochs = 10
args.batch_size = 128
args.train_samples = 1_000
train_dataset = ParticleDataset(Path(PATH + '/train_sig.hdf5'), Path(PATH + '/train_bkg.hdf5'),
        num_samples=args.train_samples, open_at_init=True)

train_loader = GraphDataLoader(dataset=train_dataset, batch_size=args.batch_size, 
        shuffle=True, collate_fn=collate_fn)
val_loader = GraphDataLoader(dataset=train_dataset, batch_size=args.batch_size, 
        shuffle=True, collate_fn=collate_fn)

#wandb_cluster_mode()

args.manifold = 'euclidean'
args.best_model_name = 'best_jets_' + args.manifold
#with wandb.init(project='loss_function', entity='office4005', config=dict(args)):
train(train_loader, val_loader, args=args)
