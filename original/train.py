import torch
import numpy as np
import os
from time import time

from torch_geometric.datasets import TUDataset, UPFD
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import DataLoader
from utils import wandb_cluster_mode, save_torch_model

import wandb


from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from data_handler import ParticleDataset

from sklearn.metrics import accuracy_score

#from models.gcn import GCN
from optimizer.radam import RiemannianAdam as RAdam
#from temp_optimizer.radam import RiemannianAdam as RAdam
from models.hgcn import HyperGNN
from manifold.poincare import PoincareBall


# loading the data
#PATH = 'data/jet_tagging/'
#train_bkg = ParticleDataset(PATH + '/train_bkg/')
#train_sig = ParticleDataset(PATH + '/train_sig/')
#valid_bkg = ParticleDataset(PATH + '/valid_bkg/')
#valid_sig = ParticleDataset(PATH + '/valid_sig/')
#test_bkg = ParticleDataset(PATH + '/test_bkg/')
#test_sig = ParticleDataset(PATH + '/test_sig/')
#
#train_dataset = ConcatDataset([train_bkg, train_sig])
#valid_dataset = ConcatDataset([valid_bkg, valid_sig])
#test_dataset = ConcatDataset([test_bkg, test_sig])
#
#train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True,
#        num_workers=40)
#valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False,
#        num_workers=40)
#test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False,
#        num_workers=40)





def train():
    model.train()
    avg_loss = []
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        avg_loss.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(avg_loss)

@torch.no_grad()
def test(loader):
    model.eval()
    prediction = []
    target = []
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        #out = global_mean_pool(out, data.batch)
        prediction.append(out.argmax(dim=1))
        target.append(data.y)
   
    prediction = [item for sublist in prediction for item in sublist]
    target = [item for sublist in target for item in sublist]

    accuracy = accuracy_score(target, prediction)
    #precision = precision_score(target, prediction)
    return accuracy#j, precision


def ROC_area(signal_eff, background_eff):
    '''Area under the ROC curve.'''
    normal_order = signal_eff.argsort()
    return np.trapz(background_eff[normal_order], signal_eff[normal_order])


#model = GCN(4, 64, 2).double()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#criterion = torch.nn.CrossEntropyLoss()
#print(model)


#wandb.config = {
#    "learning_rate": 0.01,
#    "epochs": 5,
#    "batch_size": 64,
#}

def train_script():
    """Saving the trains script as a funtion"""
    # Load UPFD dataset
    path='data/UPFD'
    dataset='politifact'
    feature='profile'
    
    train_dataset = UPFD(path, dataset, feature, 'train', ToUndirected())
    val_dataset = UPFD(path, dataset, feature, 'val', ToUndirected())
    test_dataset = UPFD(path, dataset, feature, 'test', ToUndirected())
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')
    # Initialise Model
    #model = GCN(hidden_channels=64)
    #model.to(device)
    #
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #criterion = torch.nn.CrossEntropyLoss()
    
    # Initialise the model and the optimizer 
    
    manifold = PoincareBall()
    model = HyperGNN(
            manifold, 
            train_dataset.num_node_features,
            128, 
            train_dataset.num_classes
            )#.double()
    optimizer = RAdam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    manifold = PoincareBall()
    run_config={
            'dataset': train_dataset,
            'training_graphs': len(train_dataset),
            'testing_graphs': len(test_dataset),
            'features': train_dataset.num_features,
            'classes': train_dataset.num_classes,
            'model': model,
    
            }
    print()
    print(f'Dataset: {train_dataset}:')
    print('====================')
    print(f'Number of graphs: {len(train_dataset)}')
    print(f'Number of features: {train_dataset.num_features}')
    print(f'Number of classes: {train_dataset.num_classes}')
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(model)
    
    print('Start the training')
    wandb_cluster_mode()
    wandb.init(
            project='jet_tagging_testing', 
            entity='office4005',
            config=run_config
            )
    for epoch in range(1, 301):
        init = time()
        avg_loss = train()
        fin = time() - init
        train_acc = test(train_loader)
        valid_acc = test(val_loader)
    
        wandb.log({
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "loss": avg_loss,
            'epoch': epoch,
        })
        
    
    print(f'The training has ended')
    test_acc = test(test_loader)
    print(f'Test -- Accuracy: {test_acc:.4f}')
    
    save_torch_model(model, 'chckpts/hgcn_test', 'hgcn')
    
