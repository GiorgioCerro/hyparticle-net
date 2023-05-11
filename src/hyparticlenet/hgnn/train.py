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
from hyparticlenet.hgnn.util import wandb_cluster_mode

import wandb
from omegaconf import OmegaConf, DictConfig
from rich.progress import track
from rich.pretty import pprint
from tqdm import tqdm 


def train(train_loader, val_loader, args:DictConfig):

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
        #args.embed_dim += 1
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

    # Train, store model with best accuracy on validation set
    best_accuracy = 0
    for epoch in track(range(args.epochs), 
            description="[cyan]{} Training epoch".format(experiment_name)):
        model.train()

        total_loss = 0
        init = time.time()
        for batch in train_loader:
            label = batch.label
            label = label.to(device).squeeze().long()
            num_graphs = label.shape[0]

            optimizer.zero_grad()
            logits = model(batch.batch_graph.to(device), batch.features.to(device))

            loss = loss_function(logits, label)
            loss.backward(retain_graph=True)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            total_loss += loss.item() * num_graphs
            optimizer.step()

   
        scheduler.step()

        # compute valid accuracy
        #train_acc = evaluate(args, model, train_loader)
        #val_acc, val_auc = evaluate(args, model, val_loader, return_auc=True)
        # compute training accuracy and training loss
        train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - init
        print(
                f'Epoch: {epoch:n}, ',
                f'loss: {total_loss:.3f}, '
        )

        
        #if val_acc > best_accuracy:
        #    p = Path(args.logdir)
        #    p.mkdir(parents=True, exist_ok=True)
        #    torch.save(model.state_dict(), p.joinpath(f'{args.best_model_name}.pt'))
        #    best_accuracy = val_acc
        
        # Log to wandb
        #wandb.log({
        #    'epoch': epoch,
        #    'validation_auc': val_auc,
        #    'validation_accuracy': val_acc,
        #    'training_accuracy': train_acc,
        #    'training_loss': train_loss,
        #    })
    

def evaluate(args, model, data_loader, return_auc=None):
    """Evaluate the model and return accuracy and AUC.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    soft = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        scores = np.zeros(len(data_loader.dataset))
        target = np.zeros(len(data_loader.dataset), dtype=int)
        c = 0 
        for data in data_loader:
            data = data.to(device)
            pred = soft(model(data))[:, 1]
            scores[args.batch_size * c : args.batch_size * (c+1)] = pred.cpu()
            target[args.batch_size * c : args.batch_size * (c+1)] = data.y.cpu()
            c+=1

    labels = (scores >= 0.5).astype(int)
    accuracy = accuracy_score(target, labels)
    
    fpr, tpr, threshs = roc_curve(target, scores, pos_label=1)
    eff_s = tpr
    eff_b = 1 - fpr
    auc = ROC_area(eff_s, eff_b)

    if return_auc:
        return accuracy, auc
    else:
        return accuracy


def ROC_area(signal_eff, background_eff):
    """Area under the ROC curve.
    """
    normal_order = signal_eff.argsort()
    return np.trapz(background_eff[normal_order], signal_eff[normal_order])
