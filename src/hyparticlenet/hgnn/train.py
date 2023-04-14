import os, os.path as osp
import time
import warnings

from pathlib import Path

import numpy as np
import random
import torch
from sklearn.metrics import roc_curve, accuracy_score

from hyparticlenet.hgnn.models.graph_classification import GraphClassification
from hyparticlenet.hgnn.nn.manifold import EuclideanManifold, PoincareBallManifold, LorentzManifold
from hyparticlenet.hgnn.util import wandb_cluster_mode, MeanAveragePrecision

import wandb
from omegaconf import OmegaConf, DictConfig
from rich.progress import track
from rich.pretty import pprint
from tqdm import tqdm 

HGNN_CONFIG = {
    # Hyperbolic GNN parameters
    'in_features': 5,
    'embed_dim': 5,
    'num_layers': 5,
    'num_class': 3,
    'num_centroid': 100,
    'manifold': 'poincare',
    
    # Training parameters
    'optimizer': 'adam',
    'lr': 0.001,
    'weight_decay': 0,
    'grad_clip': 1,
    'dropout': 0,
    'batch_size': 32,
    'epochs': 30,
    'alpha': 0.5, # loss weight
    'weight_init': 'xavier', # other option 'default' or empty
    'seed': 123,
    'device': 'cpu',
    'logdir': 'logs',
    'best_model_name': 'best',
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
    print(f'Using this device: {device}')

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
    #path_load = "/home/gc2c20/myproject/hyparticle-net/experiments/logs/best_jets_d2_poincare.pt" 
    #model.load_state_dict(torch.load(path_load))


    # And optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
            amsgrad=args.optimizer == 'amsgrad', weight_decay=args.weight_decay)
    lr_steps = [60, 70]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=lr_steps, gamma=0.1)

    loss_function = torch.nn.CrossEntropyLoss(reduction='mean')

    #Start Wandb
    #wandb_cluster_mode()
    #wandb.init(
    #        project='jet_tagging_testing',
    #        entity='office4005',
    #        config=dict(args),
    #        )

    # Train, store model with best accuracy on validation set
    best_accuracy = 0
    for epoch in track(range(args.epochs), 
            description="[cyan]{} Training epoch".format(experiment_name)):
        model.train()

        total_loss, total_l1, total_l2 = 0, 0, 0
        init = time.time()
        for data in train_loader:
            batch_time = time.time()
            model.zero_grad()
            data = data.to(device)
            embedding, out = model(data)

            l1 = MeanAveragePrecision(data, embedding, device)
            l2 = loss_function(out, data.y)
            loss = args.alpha * l1 + (1 - args.alpha) * l2

            loss.backward(retain_graph=True)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            total_loss += loss.item() * data.num_graphs
            total_l1 += l1.item() * data.num_graphs
            total_l2 += l2.item() * data.num_graphs
            optimizer.step()

   
        scheduler.step()

        # compute valid accuracy
        train_acc = evaluate(args, model, train_loader)
        val_acc, val_loss, val_auc = evaluate(args, model, val_loader, return_auc=True)
        # compute training accuracy and training loss
        train_loss = total_loss / len(train_loader)
        train_l1 = total_l1 / len(train_loader)
        train_l2 = total_l2 / len(train_loader)
        epoch_time = time.time() - init
        pprint(f'Epoch {epoch:n}, time {epoch_time:.3f}')
        pprint(f'Train loss {train_loss:.5f}, train acc. {train_acc:.5f}')
        pprint(f'MeanAveragePrecision {train_l1:.5f}, CrossEntropyLoss {train_l2:.5f}')
        pprint(f'Valid acc. {val_acc:.5f}, valid auc {val_auc:.5f}')
        pprint(20*'~')

        
        if val_acc > best_accuracy:
            p = Path(args.logdir)
            p.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), p.joinpath(f'{args.best_model_name}.pt'))
            best_accuracy = val_acc
        
        # Log to wandb
        wandb.log({
            'validation_auc': val_auc,
            'validation_accuracy': val_acc,
            'training_accuracy': train_acc,
            'validation_loss': val_loss,
            'training_loss': train_loss,
            'mean_average_precision': train_l1,
            'cross_entropy_loss': train_l2,
            })
    

def evaluate(args, model, data_loader, return_auc=None):
    """Evaluate the model and return accuracy and AUC.
    """
    loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    soft = torch.nn.Softmax(dim=1)
    model.eval()
    loss_temp = 0
    with torch.no_grad():
        scores = np.zeros(len(data_loader.dataset))
        target = np.zeros(len(data_loader.dataset), dtype=int)
        c = 0 
        for data in data_loader:
            data = data.to(device)
            embedding, out = model(data)
            pred = soft(out)[:, 1]
            scores[args.batch_size * c : args.batch_size * (c+1)] = pred.cpu()
            target[args.batch_size * c : args.batch_size * (c+1)] = data.y.cpu()
            c+=1

            l1_temp = MeanAveragePrecision(data, embedding, device)
            l2_temp = loss_function(out, data.y)
            _loss = l1_temp + l2_temp
            loss_temp += _loss.item() * data.num_graphs

    labels = (scores >= 0.5).astype(int)
    accuracy = accuracy_score(target, labels)
    
    fpr, tpr, threshs = roc_curve(target, scores, pos_label=1)
    eff_s = tpr
    eff_b = 1 - fpr
    auc = ROC_area(eff_s, eff_b)

    if return_auc:
        return accuracy, loss_temp / len(data_loader), auc
    else:
        return accuracy


def ROC_area(signal_eff, background_eff):
    """Area under the ROC curve.
    """
    normal_order = signal_eff.argsort()
    return np.trapz(background_eff[normal_order], signal_eff[normal_order])
