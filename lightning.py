import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torch as th
import torch.nn.functional as F
from manifold.poincare import PoincareManifold
poincare = PoincareManifold()

def distance_matrix(nodes):
    '''
    _a = (nodes[:,0][...,None] - nodes[:,0]) ** 2.
    _b = (nodes[:,1][...,None] - nodes[:,1]) ** 2.
    matrix = th.sqrt(_a + _b + 1e-8)
    '''
    matrix = th.zeros(len(nodes),len(nodes))
    for n_idx in range(len(nodes)):
        matrix[n_idx] = poincare.distance(nodes[n_idx],nodes)
    return matrix

class LitHGNN(pl.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.hgnn = model(4,64,64,2)
        self.hgnn.double()


    def training_step(self,batch,batch_idx):
        output = self.hgnn(batch)

        loss=0
        for graph_idx in th.unique(batch.batch):
            graph_mask = batch.batch == graph_idx
            _x = output[graph_mask]
            _y = batch.y[graph_mask]

            #_x = poincare._clip_vectors(_x)
            #_y = poincare._clip_vectors(_y)

            _input = distance_matrix(_x)
            _target = distance_matrix(_y)

            loss += F.mse_loss(_input,_target)

        loss /= batch.num_graphs
        self.log('training loss',loss)
        return loss


    def test_step(self,batch,batch_idx):
        output = self.hgnn(batch)

        loss=0
        for graph_idx in th.unique(batch.batch):
            graph_mask = batch.batch == graph_idx
            _x = output[graph_mask]
            _y = batch.y[graph_mask]

            _input = distance_matrix(_x)
            _target = distance_matrix(_y)

            loss += F.mse_loss(_input,_target)

        loss /= batch.num_graphs
        self.log('training loss',loss)


    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(),lr=1e-2)
        return optimizer



class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running training ...')
        return bar
