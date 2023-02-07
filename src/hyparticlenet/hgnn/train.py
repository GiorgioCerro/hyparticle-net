import os, os.path as osp
import time
import warnings
#from progressbar import progressbar

import numpy as np
import random

from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from hyparticlenet.hgnn.models.graph_classification import GraphClassification
from hyparticlenet.hgnn.nn.manifold import EuclideanManifold, PoincareBallManifold, LorentzManifold
from hyparticlenet.hgnn.util import wandb_cluster_mode

import wandb
from omegaconf import OmegaConf, DictConfig
from rich.progress import track

HGNN_CONFIG = {
    # Hyperbolic GNN parameters
    'in_features': 1000,
    'embed_dim': 5,
    'num_layers': 5,
    'num_class': 3,
    'num_centroid': 100,
    'manifold': 'poincare',
    
    # Training parameters
    'optimizer': 'adam',
    'lr': 0.01,
    'weight_decay': 0,
    'grad_clip': 1,
    'dropout': 0,
    'batch_size': 32,
    'epochs': 80,
    'weight_init': 'xavier', # other option 'default' or empty
    'seed': 123,
    'device': 'cpu',
    'logdir': 'logs',
    'best_model_name': 'best',
    
    # Dataset parameters
    'node_num_min': 100,
    'node_num_max': 500,
    'num_train': 2000,
    'num_val': 2000,
    'num_test': 2000,
    }

HGNN_CONFIG = OmegaConf.create(HGNN_CONFIG)

def train(
        train_loader,
        val_loader,
        args:DictConfig=HGNN_CONFIG
        ):

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create log directory
    #file_dir = osp.dirname(osp.realpath(__file__))

    experiment_name = 'hgnn_{}_dim{}'.format(args.manifold, args.embed_dim)
    run_time = time.strftime("%d%b%y_%H_%M", time.localtime(time.time()))


    # Additional arguments
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    #args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)
    # Select manifold
    if args.manifold == 'euclidean':
        manifold = EuclideanManifold()
    elif args.manifold == 'poincare':
        manifold = PoincareBallManifold()
    elif args.manifold == 'lorentz':
        manifold = LorentzManifold()
        args.embed_dim += 1
    else:
        manifold = EuclideanManifold()
        warnings.warn('No valid manifold was given as input, using Euclidean as default')

    # Setup model
    model = GraphClassification(args, manifold).to(device)

    # And optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=args.optimizer == 'amsgrad', weight_decay=args.weight_decay)
    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

    # Start Wandb
    wandb_cluster_mode()
    wandb.init(
            project='jet_tagging_testing',
            entity='office4005',
            config=dict(args),
            )

    # Train, store model with best accuracy on validation set
    best_accuracy = 0
    for epoch in track(range(args.epochs), description="[cyan]{} Training epoch".format(experiment_name)):
        model.train()

        total_loss = 0
        for data in train_loader:
            model.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = loss_function(out, data.y)
            loss.backward(retain_graph=True)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        val_acc = evaluate(args, model, val_loader)
        train_loss = total_loss / len(train_loader)
        print('Epoch {:n} - training loss {:.3f}, validation accuracy {:.3f}'.format(epoch, train_loss, val_acc))
        if val_acc > best_accuracy:
            p = Path(args.logdir)
            p.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), p.joinpath(f'{args.best_model_name}.pt'))
            best_accuracy = val_acc
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'training_loss': train_loss,
            'validation_accuracy': val_acc
            })


def evaluate(args, model, data_loader):
    model.eval()
    correct = 0
    for data in data_loader:
        data = data.to(args.device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(data_loader.dataset)






