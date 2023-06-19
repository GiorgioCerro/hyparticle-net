import os, os.path as osp
import time
import warnings


import numpy as np
import random
import torch
from sklearn.metrics import roc_curve, accuracy_score
from torchmetrics import MetricCollection, ROC, classification as metrics

from hyparticlenet.hgnn.models.graph_classification import GraphClassification
from hyparticlenet.hgnn.nn.manifold import EuclideanManifold, PoincareBallManifold, LorentzManifold
from hyparticlenet.hgnn.util import count_params, ROC_area

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
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    #pprint(f'Using this device: {device}')
    
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
    num_params = count_params(model)
    pprint(f'Training model with {num_params} learnable parameters.')

   
    # And optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
            amsgrad=args.optimizer == 'amsgrad', weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, momentum=0.9)

    #lr_steps = [60, 70]
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
    #        milestones=lr_steps, gamma=0.1)

    loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
    soft = torch.nn.Softmax(dim=1)
    metric_scores = MetricCollection(dict(
        accuracy = metrics.BinaryAccuracy(),
        precision = metrics.BinaryPrecision(),
        recall = metrics.BinaryRecall(),
        f1 = metrics.BinaryF1Score(),
    ))

    # Train, store model with best accuracy on validation set
    best_accuracy = 0
    for epoch in track(range(args.epochs), 
            description="[cyan]{} Training epoch".format(experiment_name)):
        model.train()

        total_loss = 0
        init = time.time()
        for graph, label in train_loader:
            label = label.to(device).squeeze().long()
            num_graphs = label.shape[0]

            optimizer.zero_grad()
            logits = model(graph.to(device))

            loss = loss_function(logits, label)
            loss.backward()

            pred = soft(logits)[:, 1]
            #pred = (pred >= 0.5).long()
            metric_scores.update(pred, label)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            total_loss += loss.item() * num_graphs
            optimizer.step()
   
        #scheduler.step()

        scores = metric_scores.compute()

        # compute valid accuracy
        val_acc, val_loss, val_auc = evaluate(model, val_loader)
        train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - init
        pprint(
            f"epoch: {epoch:n}, "
            f"loss: {train_loss:.5f}, "
            f"accuracy: {scores['accuracy'].item():.1%}, "
            f"precision: {scores['precision'].item():.1%}, "
            f"recall: {scores['recall'].item():.1%}, "
            f"f1: {scores['f1'].item():.1%}, "
            f"time: {epoch_time:.2f} "
        )

        
        if val_acc > best_accuracy:
        #    p = Path(args.logdir)
        #    p.mkdir(parents=True, exist_ok=True)
        #    pprint("Saving the best model")
        #    torch.save(model.state_dict(), p.joinpath(f'{args.best_model_name}.pt'))
            best_accuracy = val_acc
        

        # Log to wandb
        #wandb.log({
        #    'validation_auc': val_auc,
        #    'validation_accuracy': val_acc,
        #    'validation_loss': val_loss,
        #    'accuracy': scores['accuracy'].item(),
        #    'precision': scores['precision'].item(),
        #    'recall': scores['recall'].item(),
        #    'f1': scores['f1'].item(),
        #    'loss': train_loss,
        #})
    
    return best_accuracy


def evaluate(model, data_loader):
    """Evaluate the model and return accuracy, loss and AUC.
    """
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
    soft = torch.nn.Softmax(dim=1)
    model.eval()

    metric_scores = MetricCollection(dict(
        accuracy = metrics.BinaryAccuracy(),
        ROC = ROC(task="binary"),
    ))
    loss_temp = 0 
    with torch.no_grad():
        for graph, label in tqdm(data_loader):
            label = label.to(device).squeeze().long()
            num_graphs = label.shape[0]

            logits = model(graph.to(device))


            pred = soft(logits)[:, 1]
            #pred = (pred >= 0.5).long()
            metric_scores.update(pred, label)

            loss_temp += loss_function(logits, label).item() * num_graphs

    scores = metric_scores.compute()
    accuracy = scores['accuracy'].item()
    fpr, tpr, threshs = scores['ROC']
    eff_s = tpr
    eff_b = 1 - fpr
    auc = ROC_area(eff_s, eff_b)

    return accuracy, loss_temp / len(data_loader), auc

