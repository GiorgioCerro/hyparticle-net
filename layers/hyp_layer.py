import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
from torch import Tensor

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import to_dense_adj


class HyperbolicGraphConvolution(nn.Module):
    '''Hyperbolic graph convolution layer.
    '''
    def __init__(self, manifold, in_channels, out_channels,
        dropout=0., active=None):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HyperbolicLinear(manifold, in_channels, out_channels)
        self.aggregation = HyperbolicAggregation(manifold, out_channels, 
            dropout)
        self.activation = HyperbolicActivation(manifold, active)
        self.active = active

    def forward(self, x, edge_index):
        adjacency = to_dense_adj(edge_index)[0].double() 

        h = self.linear.forward(x)
        #h = self.aggregation.forward(h, adjacency)
        #if self.active:
        #    h = self.activation.forward(h)
        return h

class HyperbolicLinear(nn.Module):
    '''Hyperbolic neural networks layer.
    '''
    def __init__(self, manifold, in_channels, out_channels, bias=True):
        super(HyperbolicLinear, self).__init__()
        
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.weights = Parameter(torch.Tensor(out_channels, in_channels))
        self.reset_parameters()


    def reset_parameters(self):
        init.normal_(self.weights, mean=0., std=0.2)
        zeros(self.bias)

    def forward(self,x):
        '''
        drop_weight = F.dropout(self.weights, self.dropout,
            training = self.training)
        result = self.manifold.mobius_matvec(
            drop_weight, x)
        '''
        result = self.manifold.mobius_matvec(self.weights, x)

        if self.bias is not None:
            result = self.manifold.mobius_add(
                result, self.bias.repeat(len(result),1))

        result = self.manifold.proj(result)
        return result


class HyperbolicAggregation(nn.Module):
    '''Hyperbolic aggregation layer
    '''

    def __init__(self, manifold, in_channels, dropout, aggregation=None,
        use_attention=False):
        super(HyperbolicAggregation, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.dropout = dropout
        self.use_attention = use_attention

    def forward(self, x, adj):
        neighbours = torch.sum(adj, axis=-1).view(-1,1)
        x_tangent = self.manifold.logmap(x) 
        aggregation = torch.spmm(adj,x_tangent)  / neighbours 
        x_hyperbolic = self.manifold.proj(self.manifold.expmap(aggregation))
        return x_hyperbolic


class HyperbolicActivation(nn.Module):
    '''Hyperbolic activation layer
    '''
    def __init__(self, manifold,  activation):
        super(HyperbolicActivation, self).__init__()
        self.manifold = manifold
        self.activation = activation

    def forward(self, x):
        x_tangent = self.manifold.logmap(x)
        x_activated = self.activation(x_tangent)
        x_hyperbolic = self.manifold.proj(self.manifold.expmap(x_activated))
        return x_hyperbolic
