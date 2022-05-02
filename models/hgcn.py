import torch 
from layers.hyp_layer import HyperbolicGraphConvolution as HypLayer

activation = torch.tanh
class HyperGCN(torch.nn.Module):
    def __init__(self, manifold, _in, _h1, _h2, _out):
        super().__init__()
        self.manifold = manifold
        self.hconv1 = HypLayer(manifold, _in-1, _h1, dropout=0.3)
        #    active=activation)
        self.hconv2 = HypLayer(manifold, _h1, _h2, dropout=0.3)
        #    active=activation)
        self.hconv3 = HypLayer(manifold, _h2, _out, dropout=0.3,
            active=activation)

    def forward(self, data):
        x = self.manifold.lorentz_to_poincare(data.x)
        x = torch.tanh(x)
        edge_index = data.edge_index

        x = self.hconv1(x, edge_index)
        x = self.hconv2(x, edge_index)
        x = self.hconv3(x, edge_index)

        return x
