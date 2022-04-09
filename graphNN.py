import torch as th
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv, SAGEConv

from dataHandler import ParticleDataset
from manifold.poincare import PoincareManifold
poincare = PoincareManifold()
'''
def distance_matrix(nodes):
    _a = (nodes[:,0][...,None] - nodes[:,0]) ** 2.
    _b = (nodes[:,1][...,None] - nodes[:,1]) ** 2.
    matrix = th.sqrt(_a + _b + 1e-8)
    return matrix
'''
class GNN(th.nn.Module):
    def __init__(self,_in,_h1,_h2,_out):
        super().__init__()
        self.tan = th.nn.Tanh()
        self.conv1 = HypergraphConv(_in,_h1,dropout=0.1)
        self.conv2 = HypergraphConv(_h1,_h2,dropout=0.1)
        self.conv3 = HypergraphConv(_h2,_out)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index

        for row_id in range(len(x)):
            x[row_id] = poincare.exp_map(x[row_id],th.zeros_like(x[0]))

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        return x
