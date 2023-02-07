import torch 
from torch.nn import Linear
import torch.nn.functional as F
from layers.hyp_layer import HNNLayer, HGCNLayer
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj, add_self_loops

class HyperGNN(torch.nn.Module):
    def __init__(
            self, 
            manifold, 
            _in, 
            _h1, 
            _out):
        super().__init__()
        self.manifold = manifold
        self.hconv1 = HGCNLayer(manifold, in_features=_in, out_features=_h1,
                                dropout=0.0, act=torch.nn.LeakyReLU(0.2), 
                                local_agg=False, use_bias=True)
        self.hconv2 = HGCNLayer(manifold, in_features=_h1, out_features=_h1,
                                dropout=0.0, act=torch.nn.LeakyReLU(0.2),
                                local_agg=False, use_bias=True)
        self.hconv3 = HGCNLayer(manifold, in_features=_h1, out_features=_h1,
                                dropout=0.0, act=None,
                                local_agg=False, use_bias=True)
        self.hlin = HNNLayer(manifold, in_features=_h1, out_features=_out, 
                                dropout=0.0, use_bias=True)
        

    def forward(self, x, edge_index, batch):
        add_self_loops(edge_index)

        # map the data into the hyp space
        x = self.manifold.expmap(x)
        x = self.manifold.proj(x)

        x = self.hconv1(x, edge_index)
        x = self.hconv2(x, edge_index)
        x = self.hconv3(x, edge_index)

        x = self.manifold.logmap(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.manifold.expmap(x)
        x = self.hlin(x)

        return x



