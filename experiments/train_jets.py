import wandb
from torch.utils.data import DataLoader

from hyparticlenet.hgnn.train import train
from hyparticlenet.hgnn.util import wandb_cluster_mode

from lundnet.dgl_dataset import DGLGraphDatasetLund, collate_wrapper_tree

from omegaconf import OmegaConf
config_path = 'configs/jets_config.yaml'
args = OmegaConf.load(config_path)


import torch
NUM_THREADS = 4
torch.set_num_threads = NUM_THREADS

# Jets Data sets
PATH = '/scratch/gc2c20/data/w_tagging/'

train_dataset = DGLGraphDatasetLund(PATH+'/train_bkg/', PATH+'/train_sig/', 
                                nev=-1, n_samples=args.train_samples)
valid_dataset = DGLGraphDatasetLund(PATH+'/valid_bkg/', PATH+'/valid_sig/', 
                                nev=-1, n_samples=args.valid_samples)

collate_fn = collate_wrapper_tree
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
        shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn)


wandb_cluster_mode()

args.loss_embedding = False
args.best_model_name = 'best_jets_' + args.manifold
#with wandb.init(project='loss_function', entity='office4005', config=dict(args)):
train(train_loader, val_loader, args=args)



