import torch

import torch
from torch_geometric.utils import to_dense_adj


def dot(a, b):
    return torch.bmm(a.unsqueeze(-2), b.unsqueeze(-1)).squeeze(-1)

def atanh(x, EPS):
	values = torch.min(x, torch.Tensor([1.0 - EPS]).to(x.device))
	return 0.5 * (torch.log(1 + values + EPS) - torch.log(1 - values + EPS))

def clamp_min(x, min_value):
	t = torch.clamp(min_value - x.detach(), min=0)
	return x + t

def wandb_cluster_mode():
    """
    Get wandb key and turn wandb offline. Requires os imported?
    """
    import os
    key = os.environ.get("WANDB_KEY")
    os.environ['WANDB_API_KEY'] = key 
    os.environ['WANDB_MODE'] = 'offline'
    #os.environ['WANDB_MODE'] = 'online'


def distance_matrix(nodes):
    sq_norms = torch.sum(nodes ** 2, dim=-1, keepdim=True)
    sq_dists = sq_norms + sq_norms.transpose(0, 1) - 2 * nodes @ nodes.transpose(0, 1)
    cosh_angle = 1 + 2 * sq_dists / ((1 - sq_norms) * (1 - sq_norms.transpose(0, 1)))
    cosh_angle.clamp_min_(1 + 1e-8)
    dist = torch.arccosh(cosh_angle) + 1e-8
    dist.fill_diagonal_(1e-8)
    return dist


def MeanAveragePrecision(batch, embedding, device=torch.device('cpu')):
    '''Get the mean average precision for all the different graphs.
    '''
    losses = []
    for batch_idx in range(len(batch.y)):
        graph = batch[batch_idx]
        hyp = embedding[batch.batch == batch_idx].to(device)
    
        _dist = distance_matrix(hyp)
        radius = _dist[graph.edge_index[0], graph.edge_index[1]]
        distances = _dist[graph.edge_index[0]]

        nodes_in_circle = distances <= radius.unsqueeze(-1)
        adj = to_dense_adj(graph.edge_index)[0]
        neighbours = adj[graph.edge_index[0]]

        num = (nodes_in_circle * neighbours).sum(1)
        # -1 for excluding self-loops
        den = (nodes_in_circle.sum(1) - 1) * neighbours.sum(1)

        mAP = torch.sum(num / den) / graph.num_nodes
        losses.append(mAP)

    return - torch.mean(torch.tensor(losses))
