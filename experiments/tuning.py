import sys
import wandb
from torch_geometric.loader import DataLoader
from lundnet.pyg_dataset import DGLGraphDatasetLund

from hyparticlenet.hgnn.train import HGNN_CONFIG, train
from hyparticlenet.hgnn.nn.manifold import PoincareBallManifold
from hyparticlenet.hgnn.models.graph_classification import GraphClassification
from hyparticlenet.hgnn.util import wandb_cluster_mode

import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve

from tqdm import tqdm
import random

PATH = '/scratch/gc2c20/data/w_tagging/'

train_dataset = DGLGraphDatasetLund(PATH+'/train_bkg/', PATH+'/train_sig/', nev=-1,
                                n_sample=50000)
valid_dataset = DGLGraphDatasetLund(PATH+'/valid_bkg/', PATH+'/valid_sig/', nev=-1,
                                n_sample=7000)


def main():
    wandb_cluster_mode()
    for i in range(40):
        hp = HGNN_CONFIG
        hp.batch_size = random.randrange(24, 65)
        hp.epochs = random.randrange(20, 40)
        hp.lr = random.uniform(0.0001, 0.01)
        hp.embed_dim = random.randrange(5, 41)
        hp.num_centroid = random.randrange(2, 81) 
        hp.in_features = 5
        hp.best_accuracy = 0.5

        with wandb.init(project='sweep', entity='office4005', config=dict(hp)):
            train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, 
                    shuffle=True, num_workers=20)
            valid_loader = DataLoader(valid_dataset, batch_size=hp.batch_size, 
                    shuffle=True, num_workers=20)

            
            manifold = PoincareBallManifold()
            model = GraphClassification(hp, manifold)

            best_acc = train(train_loader, valid_loader, args=hp)
            wandb.config.update({'best_accuracy' : best_acc}, allow_val_change=True)

    
if __name__ == '__main__':
    sys.exit(main())
