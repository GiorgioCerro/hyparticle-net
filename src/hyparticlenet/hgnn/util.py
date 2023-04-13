import torch

import os.path as osp
import torch
from torch_geometric.utils import from_networkx, degree
from torch_geometric.data import InMemoryDataset

import numpy as np
import networkx as nx
from rich.progress import track
from rich.pretty import pprint

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


def sqdist(x, y):
    sq_norm_x = torch.norm(x, dim=-1) ** 2.
    sq_norm_y = torch.norm(y, dim=-1) ** 2.
    sq_norm_xy = torch.norm(x - y, dim=-1) ** 2.

    cosh_angle = 1 + 2 * sq_norm_xy / ((1 - sq_norm_x) * (1 - sq_norm_y))
    cosh_angle.clamp_min_(1 + 1e-8)
    dist = torch.arccosh(cosh_angle)
    return dist


def distance_matrix(nodes):
    length = len(nodes)
    matrix = torch.zeros((length, length))
    for n_idx in range(length):
        nd = nodes[n_idx][None, :]
        matrix[n_idx] = sqdist(nd, nodes) + 1e-8
        
    return matrix


def MeanAveragePrecision(batch, embedding):
    '''Get the mean average precision for all the different graphs.
    '''
    losses = []
    for batch_idx in range(len(batch.y)):
        data = batch[batch_idx]
        hyp = embedding[batch.batch == batch_idx]
        nodes = torch.tensor(range(data.x.shape[0]))
        edges = data.edge_index
        distances = distance_matrix(hyp)
        mAP = 0

        for node in nodes:
            # get the neighbours of a node
            neighbours = edges[1][ edges[0] == node]
            temp_mAP = 0
            for neigh in neighbours:
                # define the circle's radius
                radius = distances[node][neigh]

                # find all the nodes within the circle
                radius_mask = distances[node] <= radius
                # remove self loop
                radius_mask[node] = False
                nodes_in_circle = nodes[radius_mask]
                # count how manyy should be there
                combined = torch.cat((neighbours, nodes_in_circle))
                uniques, counts = combined.unique(return_counts=True)
                intersection = uniques[counts > 1]
                temp_mAP  += len(intersection) / len(nodes_in_circle)

            mAP += temp_mAP / len(neighbours) 

        mAP /= len(nodes)
        losses.append(mAP)
    
    loss = torch.mean(torch.tensor(losses))
    return loss
