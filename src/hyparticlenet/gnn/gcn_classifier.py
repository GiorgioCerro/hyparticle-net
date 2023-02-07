import numpy as np
import torch
from torch_geometric.datasets import TUDataset, UPFD
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import DataLoader
import wandb
from utils import wandb_cluster_mode
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from rich.progress import track
from pathlib import Path
import os


'''
Reference:
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/upfd.py
'''



SEED = np.random.randint(40,18329)

def train():
    model.train()
    avg_loss = []
    for data in train_loader:  # Iterate in batches over the training dataset.
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        avg_loss.append(loss.detach().numpy())
    return np.mean(avg_loss)

@torch.no_grad()
def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(SEED)
        self.conv1 = GCNConv(train_dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, train_dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def train_script():
    """Temp function to save the train script"""
    ## Load MUTAG dataset 
    #dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    #
    #torch.manual_seed(SEED)
    #dataset = dataset.shuffle()
    #train_dataset = dataset[:150]
    #test_dataset = dataset[150:]
    #
    ## Initialise Loaders
    #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
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
    model = GCN(hidden_channels=64)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    
    run_config={
            'dataset': train_dataset,
            'training_graphs': len(train_dataset),
            'testing_graphs': len(test_dataset),
            'features': train_dataset.num_features,
            'classes': train_dataset.num_classes,
            'seed': SEED,
            'model': model
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
    
    wandb_cluster_mode()
    wandb.init(
            project='jet_tagging_testing', 
            entity='office4005',
            config=run_config
            )
    
    for epoch in track(range(1, 301), description="Training..."):
        avg_loss = train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        valid_acc = test(val_loader)
    
        wandb.log({
            "train_acc": train_acc,
            "test_acc": test_acc,
            "valid_acc": valid_acc,
            "loss": avg_loss,
            "epoch": epoch
            })
    
    chckpt = Path('chckpts/saved_gcn_model_tree')
    chckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), chckpt)
