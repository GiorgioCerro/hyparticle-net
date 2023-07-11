import typing as ty
import warnings

import torch.nn as nn

from hyparticlenet.centroid_distance import CentroidDistance
from hyparticlenet.manifold import EuclideanManifold, LorentzManifold, PoincareBallManifold
from hyparticlenet.manifold_edgeconv import ManifoldEdgeConv


class HyperbolicGNN(nn.Module):
    r"""
    Graph classification network based on Hyperbolic GNN paper.

    Args:
        args (dict): Arguments supplied to network.
        manifold (Manifold): Manifold to use as embedding space.
            (default: :obj:`EuclideanManifold`)
    """

    def __init__(
        self, 
        input_dims: int = 5, 
        conv_params: ty.List = [[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]],
        #conv_params: ty.List = [[32, 32], [64, 64], [128, 128]],
        num_centroid: int = 100,
        num_class: int = 2,
        manifold: str = "euclidean",
    ) -> None:
        super(HyperbolicGNN, self).__init__()
        if manifold == "euclidean":
            self.manifold = EuclideanManifold()
        elif manifold == "poincare":
            self.manifold = PoincareBallManifold()
        elif manifold == "lorentz":
            self.manifold = LorentzManifold()
        else: 
            self.manifold = EuclideanManifold()
            warnings.warn("No valid manifold - using Euclidean as default")
        
        self.bn_fts = nn.BatchNorm1d(input_dims)

        self.edge_convs = nn.ModuleList()
        for idx, channels in enumerate(conv_params):
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][-1]
            self.edge_convs.append(ManifoldEdgeConv(in_feat=in_feat, 
                                                    out_feats=channels))

        self.centroid_distance = CentroidDistance(embed_dim=conv_params[-1][-1], 
                                                num_centroid=num_centroid,
                                                manifold=self.manifold)

        self.fc = nn.Linear(num_centroid, num_class)

    def forward(self, batch_graph):
        x = self.bn_fts(batch_graph.ndata['features'])
        for idx, conv in enumerate(self.edge_convs):
            x = conv(batch_graph, x)
        centroid_dist = self.centroid_distance(batch_graph, x)
        return self.fc(centroid_dist)
