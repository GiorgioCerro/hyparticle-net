import torch
import torch.nn as nn
from dgl.nn import AvgPooling
from hyparticlenet.manifold import Manifold, EuclideanManifold


class CentroidDistance(nn.Module):
    r"""Places `num_centroid` centroids in the embedding space and computes the distances
    to each centroid from all nodes.

    Args:
        num_centroid (int): Number of centroids to place.
        embed_dim (int): Dimensionality of the embedding space.
        manifold (Manifold, optional): Manifold of embedding space.
            (default: :obj:`EuclideanManifold`)
        weight_init (str, optional): If set to `xavier`, initializes weights
            with xavier initialization. Otherwise, uses default initialization for centroid placement.
            (default: :obj:`None`)
    """

    def __init__(
        self, 
        embed_dim: int,
        num_centroid: int, 
        manifold: Manifold = EuclideanManifold(), 
    ) -> None:
        super(CentroidDistance, self).__init__()
        self.embed_dim = embed_dim
        self.num_centroid = num_centroid
        self.manifold = manifold

        self.centroid_embedding = nn.Embedding(num_centroid, embed_dim, 
                                        sparse=False, scale_grad_by_freq=False)
        self.avgpool = AvgPooling()

    def forward(self, graph, x):
        #if batch is None:
        #    batch = x.new_zeros(x.size(0)).long()
        x = self.manifold.exp(x)
        num_nodes = x.size(0)
        x = x.unsqueeze(1).expand(-1, self.num_centroid, -1).contiguous().view(-1, self.embed_dim)
        centroids = self.manifold.exp(self.centroid_embedding(torch.arange(self.num_centroid, device=x.device)))
        centroids = centroids.unsqueeze(0).expand(num_nodes, -1, -1).contiguous().view(-1, self.embed_dim)
        dist_x_centroids = self.manifold.dist(x, centroids).view(num_nodes, self.num_centroid)
        graph_centroid_dist = self.avgpool(graph, dist_x_centroids)
        return graph_centroid_dist

