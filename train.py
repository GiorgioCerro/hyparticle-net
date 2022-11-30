import torch
import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from time import time

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv, EdgeConv, GATConv

from tqdm import tqdm
from data_handler import ParticleDataset

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import roc_auc_score, roc_curve
#from optimizer.radam import RiemannianAdam as RAdam
from temp_optimizer.radam import RiemannianAdam as RAdam
from models.hgcn import HyperGNN
from manifold.poincare import PoincareBall
manifold = PoincareBall()


#loading the data
train_bkg = ParticleDataset('data/compare_tree/train_bkg/')
train_sig = ParticleDataset('data/compare_tree/train_sig/')
valid_bkg = ParticleDataset('data/compare_tree/valid_bkg/')
valid_sig = ParticleDataset('data/compare_tree/valid_sig/')
test_bkg = ParticleDataset('data/compare_tree/test_bkg/')
test_sig = ParticleDataset('data/compare_tree/test_sig/')

train_dataset = torch.utils.data.ConcatDataset([train_bkg, train_sig])
valid_dataset = torch.utils.data.ConcatDataset([valid_bkg, valid_sig])
test_dataset = torch.utils.data.ConcatDataset([test_bkg, test_sig])

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True,
        num_workers=40)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=True,
        num_workers=40)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True,
        num_workers=40)


class GCN(torch.nn.Module):
    def __init__(self, 
            in_channels,
            hidden_channels,
            out_channels):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        #self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        #x = x.relu()
        #x = self.conv4(x, edge_index)
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


#model = HyperGNN(manifold, 4, 64, 2)#.double()
#optimizer = RAdam(model.parameters(), lr=0.03, weight_decay=5e-4)

model = GCN(3, 64, 2).double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
print(model)


criterion = torch.nn.CrossEntropyLoss()

import numpy as np
def train():
    model.train()
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


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
    return accuracy


def ROC_area(signal_eff, background_eff):
    '''Area under the ROC curve.'''
    normal_order = signal_eff.argsort()
    return np.trapz(background_eff[normal_order], signal_eff[normal_order])


print('Start the training')
for epoch in range(1, 20):
    init = time()
    train()
    fin = time() - init
    train_acc = test(train_loader)
    valid_acc = test(valid_loader)
    print(f'Epoch: {epoch:03d} -- time: {fin:.2f}')
    print(f'Training acc.: {train_acc:.4f} -- Validation acc.: {valid_acc:.4f}')
    print(50*'~')
    if valid_acc > 0.7:
        break

print(f'The training has ended')
test_acc = test(test_loader)
print(f'Test -- Accuracy: {test_acc:.4f}')
torch.save(model, 'save_model/test.pt')

