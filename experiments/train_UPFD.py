from torch_geometric.loader import DataLoader
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected
import torch_geometric.transforms as T


import os
import os.path as osp

from hgnn.train import train
from hgnn.train import HGNN_CONFIG
from hgnn.util import SyntheticGraphs

# Modify config
hp = HGNN_CONFIG
hp.logdir = 'logs'
hp.best_model_name = 'best_upfd'
hp.num_class = 2
hp.in_features = 10
hp.epochs = 300
hp.seed = 234
hp.embed_dim = 2

# Load UPFD dataset
path='/scratch/mjad1g20/hyper/hyparticle-net/data/UPFD'
dataset='politifact'
feature='profile'

train_dataset = UPFD(path, dataset, feature, 'train', ToUndirected())
val_dataset = UPFD(path, dataset, feature, 'val', ToUndirected())
test_dataset = UPFD(path, dataset, feature, 'test', ToUndirected())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define data set  
#dataset_root = 'data/SyntheticGraphs'
#transform = T.Compose((
#    T.ToUndirected(),
#    T.OneHotDegree(hp.in_features - 1, cat=False)
#))
#train_dataset = SyntheticGraphs(dataset_root, split='train', transform=transform, node_num=(hp.node_num_min, hp.node_num_max), num_train=hp.num_train, num_val=hp.num_val,  num_test=hp.num_test)
#val_dataset = SyntheticGraphs(dataset_root, split='val', transform=transform, node_num=(hp.node_num_min, hp.node_num_max), num_train=hp.num_train, num_val=hp.num_val, num_test=hp.num_test)
#
#train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=False)
#val_loader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False, drop_last=False)



# Train
train(
    train_loader,
    val_loader,
    args=hp
    )

