from torch_geometric.loader import DataLoader
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected
import torch_geometric.transforms as T


import os
import os.path as osp

from hyparticlenet.hgnn.train import train
from hyparticlenet.hgnn.train import HGNN_CONFIG
from hyparticlenet.hgnn.util import SyntheticGraphs

# Jets modules import 
from hyparticlenet.data_handler import ParticleDataset
from torch.utils.data import ConcatDataset

# Modify config
hp = HGNN_CONFIG
hp.logdir = 'logs'
hp.best_model_name = 'best_jets_d2_euclidean'
hp.num_class = 2
hp.in_features = 4
hp.epochs = 300
hp.seed = 234
hp.embed_dim = 2
hp.manifold = 'euclidean'

# Jets Data sets
PATH = '/home/mjad1g20/office_share/giorgio/jet_tagging'
train_bkg = ParticleDataset(PATH+'/train_bkg/')
train_sig = ParticleDataset(PATH+'/train_sig/')
valid_bkg = ParticleDataset(PATH+'/valid_bkg/')
valid_sig = ParticleDataset(PATH+'/valid_sig/')
test_bkg = ParticleDataset(PATH+'/test_bkg/')
test_sig = ParticleDataset(PATH+'/test_sig/')
train_dataset = ConcatDataset([train_bkg, train_sig])
valid_dataset = ConcatDataset([valid_bkg, valid_sig])
test_dataset = ConcatDataset([test_bkg, test_sig])
train_loader = DataLoader(dataset=train_dataset, batch_size=hp.batch_size, shuffle=True)
val_loader = DataLoader(dataset=valid_dataset, batch_size=hp.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=hp.batch_size, shuffle=True)

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

