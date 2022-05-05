import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
from torch import Tensor

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import to_dense_adj, add_self_loops


class HyperbolicGraphConvolution(nn.Module):
    '''Hyperbolic graph convolution layer.
    '''
    def __init__(self, manifold, in_channels, out_channels,
        dropout=0., alpha=0.2, self_loops=True):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HyperbolicLinear(manifold, in_channels, out_channels)
        self.attention = HyperbolicAttention(manifold, out_channels, 
            dropout, alpha)
        #self.activation = HyperbolicActivation(manifold, active)
        self.self_loops = self_loops

    def forward(self, x, edge_index):
        if self.self_loops:
            edge_index = add_self_loops(edge_index)[0]
        adjacency = to_dense_adj(edge_index)[0].double() 

        h = self.linear.forward(x)
        #h = self.attention.forward(h, adjacency)
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


class HyperbolicAttention(nn.Module):
    '''Hyperbolic aggregation layer
    '''

    def __init__(self, manifold, in_channels, dropout, alpha, concat=True):
        super(HyperbolicAttention, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.a = nn.Parameter(torch.empty(size=(2*in_channels, 1)))
        init.normal_(self.a, mean=0., std=0.2)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap(x) 
        e = self._prepare_attentional_mechanism_input(x_tangent)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        x_tangent_prime = torch.matmul(attention, x_tangent)
        
        if self.concat:
            x_tangent_prime = F.elu(x_tangent_prime)

        x_hyperbolic = self.manifold.proj(
            self.manifold.expmap(x_tangent_prime))
        return x_hyperbolic

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.in_channels, :])
        Wh2 = torch.matmul(Wh, self.a[self.in_channels:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


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
