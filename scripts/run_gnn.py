import torch as th
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from graphNN import GNN
from time import time
from tqdm import tqdm

from dataHandler import ParticleDataset

def distance_matrix(nodes):
    _a = (nodes[:,0][...,None] - nodes[:,0]) ** 2.
    _b = (nodes[:,1][...,None] - nodes[:,1]) ** 2.
    matrix = th.sqrt( _a + _b + 1e-8) 
    return matrix

dataset = ParticleDataset('data/hz_train.hdf5','signal')
data_batches = DataLoader(dataset, batch_size=4, shuffle=True)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = GNN(4,64,2).to(device)
model.double()

optimizer = th.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(3)):
    loss = []
    t = time()
    for data in data_batches:
        optimizer.zero_grad()
        data.to(device)
        output = model(data)
        
        loss_temp = 0
        for graph_idx in th.unique(data.batch):
            graph_mask = data.batch == graph_idx
            _x = output[graph_mask]
            _y = data.y[graph_mask]
            
            _input = distance_matrix(_x)
            _target = distance_matrix(_y)

            loss_temp += F.mse_loss(_input,_target)

        loss.append(loss_temp)
        loss_temp.backward()
        optimizer.step()

    loss = th.tensor(loss)
    print(f'epoch: {epoch}, mean loss: {th.mean(loss.clone().detach())},\
            time: {round(time() - t, 2)}')

th.save(model,'model_scripted.pt')
