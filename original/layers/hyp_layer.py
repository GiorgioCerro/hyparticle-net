import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
from torch import Tensor

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import to_dense_adj, add_self_loops

#from temp_layers.att_layers import DenseAtt


class HNNLayer(nn.Module):
    '''
    Hyperbolic neural networks layer.
    '''
    def __init__(
            self, 
            manifold, 
            in_features: int,
            out_features: int,
            dropout: float=0.,
            act=None,
            use_bias: bool=True):
        super(HNNLayer, self).__init__()
        
        self.act = act
        self.linear = HypLinear(manifold, in_features, out_features, dropout, 
                                use_bias)
        if self.act:
            self.hyp_act = HypAct(manifold, act)

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear.forward(x)
        if self.act:
            h = self.hyp_act.forward(h)
        return h


class HGCNLayer(nn.Module):
    #Hyperbolic graph convolution layer.
    def __init__(
            self, 
            manifold, 
            in_features: int, 
            out_features: int,
            dropout: float=0., 
            act=None,
            use_bias: bool=True,
            local_agg=None):
        super(HGCNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, dropout,
                                use_bias)
        self.agg = HypAgg(manifold, out_features, dropout, local_agg)
        self.act = act
        if self.act:
            self.act = HypAct(manifold, act)

    def forward(self, x, edge_index):
        adjacency = to_dense_adj(edge_index)[0]#.double() 

        h = self.linear.forward(x)
        h = self.agg.forward(h, adjacency)
        if self.act:
            h = self.act.forward(h)
        return h


class HypLinear(nn.Module):
    '''Hyperbolic neural networks layer.
    '''
    def __init__(
            self, 
            manifold, 
            in_channels: int,
            out_channels: int, 
            dropout: float=0., 
            use_bias: bool=True):
        super(HypLinear, self).__init__()
        
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_bias = use_bias
        
        if self.use_bias:
            self.bias = Parameter(torch.zeros(out_channels))
        #else:
        #    self.register_parameter('bias', None)
        self.weights = Parameter(torch.Tensor(out_channels, in_channels))
        self.reset_parameters()

        #_u, _s, _v = torch.svd(self.weights)
        #_check = _s > (1 - 1e-4)
        #_s[_check] = torch.rand(len(_s[_check]))
        #_w = torch.mm(torch.mm(_u, torch.diag(_s)), _v.t())
        self.weights = Parameter(manifold.proj(self.weights, 0.5))


    def reset_parameters(self):
        #init.normal_(self.weights, mean=0., std=0.2)
        init.xavier_uniform_(self.weights, gain = 1.4)
        #zeros(self.bias)

    def forward(self, x):
        drop_weight = F.dropout(self.weights, self.dropout,
                                training=self.training)
        result = self.manifold.mobius_matvec(drop_weight, x)
        #result = self.manifold.mobius_matvec(self.weights, x)

        if self.use_bias:
            bias = self.manifold.proj(self.bias.view(1, -1))
            hyp_bias = self.manifold.expmap(bias)
            hyp_bias = self.manifold.proj(hyp_bias)
            result = self.manifold.mobius_add(result, hyp_bias)

        result = self.manifold.proj(result)
        return result

'''
class HyperbolicAttention(nn.Module):
    #Hyperbolic aggregation layer

    def __init__(self, manifold, in_channels, dropout, alpha, concat=True):
        super(HyperbolicAttention, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.a = nn.Parameter(torch.empty(size=(2*in_channels, 1)))
        init.normal_(self.a, mean=0., std=0.2)
        self.a = nn.Parameter(manifold.proj(self.a, 0.9))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap(x) 
        e = self._prepare_attentional_mechanism_input(x_tangent)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        x_tangent_prime = torch.matmul(attention, x_tangent)
        
        #if self.concat:
        #    x_tangent_prime = F.elu(x_tangent_prime)

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
'''

class HypAct(nn.Module):
    '''
    Hyperbolic activation layer
    '''
    def __init__(self, manifold,  act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.act = act

    def forward(self, x):
        x_tangent = self.manifold.logmap(x)
        x_activated = self.act(x_tangent)
        x_hyperbolic = self.manifold.proj(self.manifold.expmap(x_activated))
        return x_hyperbolic


class HypAgg(nn.Module):
    def __init__(self, manifold, in_features, dropout, use_att=None, 
                    local_agg=None):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap(x)
        '''
        if self.local_agg:
            x_local_tangent = []
            for i in range(x.size(0)):
                x_local_tangent.append(self.manifold.logmap(x[i], x))
            x_local_tangent = torch.stack(x_local_tangent, dim=0)
            support_t = torch.mean(x_local_tangent, dim=1)
            #support_t = torch.max(x_local_tangent, dim=1)[0]
            output = self.manifold.proj(self.manifold.expmap(x, support_t))
            return output
        '''
        if self.use_att:
            adj_att = self.att(x_tangent, adj)
            support_t = torch.matmul(adj_att, x_tangent)

        else:
            support_t = torch.sparse.mm(adj, x_tangent)

        result = self.manifold.proj(self.manifold.expmap(support_t))
        return result

'''
class HyperbolicLocalAgg(nn.Module):
    def __init__(self, manifold, in_channels):
        super(HyperbolicLocalAgg, self).__init__()
        self.manifold = manifold
        self.in_channels = in_channels

    def forward(self, x, adj):
        x_local = []
        adj_mask = adj>0
        for k in range(len(x)):
            x_local.append(self.manifold.expmap(torch.sum(
                self.manifold.logmap(
                    x[adj_mask[k]], x[k]), axis=0), x[k]).view(1,-1))
        x_local_agg = torch.cat(x_local)
        return x_local_agg
'''
