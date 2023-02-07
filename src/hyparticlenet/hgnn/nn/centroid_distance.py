import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from hyparticlenet.hgnn.nn.manifold import Manifold, EuclideanManifold


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

    def __init__(self, num_centroid, embed_dim, manifold: Manifold = EuclideanManifold(), weight_init = None):
        super(CentroidDistance, self).__init__()
        self.num_centroid = num_centroid
        self.embed_dim = embed_dim
        self.manifold = manifold

        self.centroid_embedding = nn.Embedding(num_centroid, embed_dim, sparse=False, scale_grad_by_freq=False)
        if weight_init and weight_init == 'xavier':
            nn.init.xavier_uniform_(self.centroid_embedding.weight.data)

    def forward(self, x, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0)).long()
        num_nodes = x.size(0)
        x = x.unsqueeze(1).expand(-1, self.num_centroid, -1).contiguous().view(-1, self.embed_dim)
        centroids = self.manifold.exp(self.centroid_embedding(torch.arange(self.num_centroid, device=x.device)))
        centroids = centroids.unsqueeze(0).expand(num_nodes, -1, -1).contiguous().view(-1, self.embed_dim)
        dist_x_centroids = self.manifold.dist(x, centroids).view(num_nodes, self.num_centroid)
        graph_centroid_dist = global_mean_pool(dist_x_centroids, batch)
        return graph_centroid_dist

