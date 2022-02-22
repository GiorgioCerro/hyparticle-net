import torch
import torch.nn.function as F
from torch_geometric.data import Data
from torch_geometri.nn import SAGEConv

class geNN:
    # geNN: graph embedding NN
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(data.num_node_features,16)
        self.conv2 = SAGEConv(16,2)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x,edge_index)
        x = self.conv2(x,edge_index)

        return x

    
