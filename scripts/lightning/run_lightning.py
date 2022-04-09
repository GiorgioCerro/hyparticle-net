import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from torch_geometric.loader import DataLoader
import torch

import os, sys, inspect
currentdir = os.path.dirname( \
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from graphNN import GNN
from dataHandler import ParticleDataset
from lightning import LitHGNN, LitProgressBar

train = ParticleDataset('../../data/test','background')
train = DataLoader(train, batch_size=32, shuffle=True,num_workers=128)

#test = ParticleDataset('../data/hz_test.hdf5','signal')
#test = DataLoader(test, batch_size=128)#,num_workers=1)

hgnn = LitHGNN(GNN)
bar = LitProgressBar()

trainer = pl.Trainer(
        #gpus=4,
        #strategy='ddp',#[DDPPlugin()],
        #accelerator='gpu',
        #devices=2,
        #num_nodes=2,
        max_epochs=2,
        log_every_n_steps=2,
        callbacks=[bar])

trainer.fit(hgnn,train)
#trainer.save_checkpoint('model.ckpt')

#trainer.test(model=hgnn,dataloaders=test,ckpt_path='model.ckpt')
