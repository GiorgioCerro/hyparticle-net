import torch 
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor

from torch_geometric.nn.inits import glorot, zeros


class HyperbolicGraphConvolution(nn.Module):
    '''Hyperbolic graph convolution layer.
    '''
    def __init__(self, manifold, in_channels, out_channels, dropout,
        activation, u):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HyperbolicLinear(manifold, in_channels, out_channels,
            dropout)
        self.aggregation = HyperbolicAggregation(manifold, out_channels, 
            dropout)
        self.activation = HyperbolicActivation(manifold, activation)

    def forward(self, data):
        x, edge_index = data
        h = self.linear.forward(x)
        h = self.aggregation.forward(h, adjacency)
        h = self.activation.forward(h)
        output = h, edge_index
        return output

class HyperbolicLinear(nn.Module):
    '''Hyperbolic neural networks layer.
    '''
    def __init__(self, manifold, in_channels, out_channels, 
        use_attention=False, heads=1, concat=True, negative_slope=0.2,
        dropout=0, use_bias=True):
        super(HyperbolicLayer, self).__init__()
        
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_bias = True
        
        self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = Parameter(torch.Tensor(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weights)
        zeros(self.bias)

    def forward(self,x):
        drop_weight = F.dropout(self.weight, self.dropout,
            training = self.training)
        result = self.manifold.matrix_vector_multiplication(
            drop_weight, x)
        
        if self.use_bias:
            result = self.manifold.bias_translation(result, self.bias)

        return result


class HyperbolicAggregation(nn.Module):
    '''Hyperbolic aggregation layer
    '''

    def __init__(self, manifold, in_channels, dropout, use_attention=False,
        aggregation):
        super(HyperbolicAggregation, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.dropout = dropout
        self.use_attention = use_attention

    def forward(self, x, adj):
        x_tangent = self.manifold.log_map(x) 
        aggregation = torch.spmm(adj,x_tangent)
        x_hyperbolic = self.manifold.exp_map(aggregation)
        return x_hyperbolic


class HyperbolicActivation(nn.Module):
    '''Hyperbolic activation layer
    '''
    def __init__(self, manifold,  activation):
        super(HyperbolicActivation, self).__init__()
        self.manifold = manifold
        self.activation = activation

    def forward(self, x):
        x_tangent = self.manifold.log_map(x)
        x_activated = self.activation(x_tangent)
        x_hyperbolic = self.manifold.exp_map(x_activated)
        return x_hyperbolic
