import torch
import numpy as np
import os
from time import time

import wandb
wandb_key = os.environ.get('WANDB_KEY')
cluster_mode = True
if cluster_mode:
    os.environ['WANDB_API_KEY'] = wandb_key
    os.environ['WANDB_MODE'] = 'offline'

wandb.init(project='jet_tagging', entity='office4005')

from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from data_handler import ParticleDataset

from sklearn.metrics import accuracy_score

from models.gcn import GCN
#from optimizer.radam import RiemannianAdam as RAdam
#from temp_optimizer.radam import RiemannianAdam as RAdam
#from models.hgcn import HyperGNN
#from manifold.poincare import PoincareBall
#manifold = PoincareBall()


# loading the data
PATH = 'data/jet_tagging/'
train_bkg = ParticleDataset(PATH + '/train_bkg/')
train_sig = ParticleDataset(PATH + '/train_sig/')
valid_bkg = ParticleDataset(PATH + '/valid_bkg/')
valid_sig = ParticleDataset(PATH + '/valid_sig/')
test_bkg = ParticleDataset(PATH + '/test_bkg/')
test_sig = ParticleDataset(PATH + '/test_sig/')

train_dataset = ConcatDataset([train_bkg, train_sig])
valid_dataset = ConcatDataset([valid_bkg, valid_sig])
test_dataset = ConcatDataset([test_bkg, test_sig])

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True,
        num_workers=40)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False,
        num_workers=40)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False,
        num_workers=40)


# initialise the model and the optimizer 
#model = HyperGNN(manifold, 4, 64, 2)#.double()
#optimizer = RAdam(model.parameters(), lr=0.01, weight_decay=5e-4)

model = GCN(4, 64, 2).double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
print(model)


def train():
    model.train()
    avg_loss = []
    # I CHANGED THIS!!! REMEMEBR TO CHANGE IT BACK TO TRAIN LOADER
    for data in valid_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        avg_loss.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(avg_loss)


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


wandb.config = {
    "learning_rate": 0.001,
    "epochs": 5,
    "batch_size": 64,
}

print('Start the training')
for epoch in range(1, 5):
    init = time()
    avg_loss = train()
    fin = time() - init
    train_acc = test(train_loader)
    valid_acc = test(valid_loader)

    wandb.log({
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "loss": avg_loss,
    })
    

print(f'The training has ended')
test_acc = test(test_loader)
print(f'Test -- Accuracy: {test_acc:.4f}')
torch.save(model, 'save_model/test.pt')

