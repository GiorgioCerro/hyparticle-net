import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from torch_geometric.loader import DataLoader
import torch

import os, sys, inspect
currentdir = os.path.dirname( \
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

#from graphNN import GNN
from models.hgcn import HyperGCN
from dataHandler import ParticleDataset
from lightning import LitHGCN, LitProgressBar
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger

#preparing the data
train = ParticleDataset('../../data/background','background')
#use 20% of training data for validation
train_set_size = int(len(train) * 0.8)
valid_set_size = len(train) - train_set_size
#split the train set
seed = torch.Generator().manual_seed(42)
train_set, valid_set = torch.utils.data.random_split(
    train, [train_set_size, valid_set_size], generator=seed)
train_set = DataLoader(train_set, batch_size=32, num_workers=2)
valid_set = DataLoader(valid_set, batch_size=32, num_workers=2)

test = ParticleDataset('../data/hz_test.hdf5','signal')
test = DataLoader(test, batch_size=4, num_workers=1)


hyper_gcn = LitHGCN(HyperGCN)
bar = LitProgressBar()
profiler = AdvancedProfiler(dirpath='.', filename='perf_logs')
logger = TensorBoardLogger('tb_logs', name='my_model')

trainer = pl.Trainer(
        logger=logger,
        #profiler=profiler,
        gpus=4,
        strategy='ddp',#[DDPPlugin()],
        #accelerator='cpu',
        #devices=2,
        #num_nodes=2,
        max_epochs=3,
        log_every_n_steps=1,
        callbacks=[bar])

trainer.fit(hyper_gcn, train_set, valid_set)
#trainer.save_checkpoint('model.ckpt')

#trainer.test(model=hgnn,dataloaders=test,ckpt_path='model.ckpt')
