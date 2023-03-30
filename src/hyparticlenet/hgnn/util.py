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

class SyntheticGraphs(InMemoryDataset):
    r"""Synthetic graph dataset from the `"Hyperbolic Graph Neural networks"
    <https://arxiv.org/pdf/1910.12892.pdf>` paper, containing graphs generated with
    Erdos-Renyi, Watts-Strogatz, and Barabasi-Albert graph generation algorithms.
    Each graph is labelled with the algorithm that was used to generate the graph (0, 1, 2)
    and contains 100-500 nodes by default.
    Args:
        root (str): Root folder of the dataset.
        split (str, optional): Whether to use the train, val, or test split.
            (default: 'train') 
        node_num (tuple, optional): The range used to determine the number of nodes in each graph.
            (default: :obj:`(100, 200)`)
        num_train (int, optional): The number of graphs in the train set.
            (default: 2000)
        num_val (int, optional): The number of graphs in the validation set.
            (default: 2000)
        num_test (int, optional): The number of graphs in the test set.
            (default: 2000)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional):  A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed during processing.
            (default: :obj:`None`)
        pre_filter (callable, optional):  A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a filtered
            version. The data object will be filtered during processing.
            (default: :obj:`None`)
    """

    def __init__(self, root, split='train',
                 node_num=(100, 500), num_train=2000, num_val=2000, num_test=2000,
                 transform=None, pre_transform=None, pre_filter=None):
        self.node_num = node_num
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        super(SyntheticGraphs, self).__init__(root, transform, pre_transform, pre_filter)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        else:
            path = self.processed_paths[2]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        return

    def process(self):
        pprint("Train")
        torch.save(self.generate_graphs(self.num_train), self.processed_paths[0])
        pprint("Validation")
        torch.save(self.generate_graphs(self.num_val), self.processed_paths[1])
        pprint("Test")
        torch.save(self.generate_graphs(self.num_test), self.processed_paths[2])

    def generate_graphs(self, num_graphs):
        data_list = []
        for i in track(range(num_graphs), description='[red]Generating graphs: erdos_renyi'):
            num_node = np.random.randint(*self.node_num)
            graph = from_networkx(nx.erdos_renyi_graph(num_node, np.random.uniform(0.1, 1)))
            graph.y = 0
            data_list.append(graph)

        for i in track(range(num_graphs), description='[green]Generating graphs: small_world'):
            num_node = np.random.randint(*self.node_num)
            graph = from_networkx(nx.watts_strogatz_graph(num_node, np.random.randint(low=2, high=100), np.random.uniform(0.1, 1)))
            graph.y = 1
            data_list.append(graph)


        for i in track(range(num_graphs), description='[cyan]Generating graphs: barabasi_albert'):
            num_node = np.random.randint(*self.node_num)
            graph = from_networkx(nx.barabasi_albert_graph(num_node, np.random.randint(low=2, high=100)))
            graph.y = 2
            data_list.append(graph)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)
