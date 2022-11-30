import torch 
from torch.nn import Linear
import torch.nn.functional as F
from layers.hyp_layer import HNNLayer, HGCNLayer
from temp_layers.hyp_layers import HyperbolicGraphConvolution as HGC
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
        '''
        self.lin1 = HGC(manifold, _in, _h1, 1, 1, 0.5, torch.nn.ReLU(),
                True, False, False)
        self.lin2 = HGC(manifold, _h1, _out, 1, 1, 0.5, torch.nn.ReLU(),
                True, False, False)
        '''
        self.lin1 = HGCNLayer(manifold, in_features=_in, out_features=_h1,
                                dropout=0.0, act=torch.nn.LeakyReLU(0.2), 
                                local_agg=False, use_bias=True)
        self.lin2 = HGCNLayer(manifold, in_features=_h1, out_features=_h1,
                                dropout=0.0, act=torch.nn.LeakyReLU(0.2),
                                local_agg=False, use_bias=True)
        self.lin3 = HGCNLayer(manifold, in_features=_h1, out_features=_h1,
                                dropout=0.0, act=None,#torch.nn.LeakyReLU(0.2),
                                local_agg=False, use_bias=True)
        self.lin4 = HNNLayer(manifold, in_features=_h1, out_features=_out, 
                                dropout=0.0, use_bias=True)
        #self.lin4 = Linear(_h1, _out)
        

    def forward(self, x, edge_index, batch):
        add_self_loops(edge_index)
        # 1. Obtain node embeddings
        x = self.manifold.expmap(x)
        x = self.manifold.proj(x)

        x = self.lin1(x, edge_index)
        x = self.lin2(x, edge_index)
        x = self.lin3(x, edge_index)

        x = self.manifold.logmap(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.manifold.expmap(x)
        x = self.lin4(x)

        '''
        adj = to_dense_adj(edge_index)[0]
        input = (x, adj) 
        input = self.lin1(input)
        x = self.lin2(input)[0]
        '''
        return x



