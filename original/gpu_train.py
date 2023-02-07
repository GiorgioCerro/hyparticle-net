#import numpy as np
import torch
#from torch_geometric.datasets import TUDataset
#from torch_geometric.loader import DataLoader
#import wandb
#from utils import wandb_cluster_mode
#from torch.nn import Linear
#import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
#from torch_geometric.nn import global_mean_pool
#from rich.progress import track
#from pathlib import Path
#
#wandb_cluster_mode()
#
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')


