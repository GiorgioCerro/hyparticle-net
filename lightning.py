import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torch.nn.functional as F
import torch

from optimizer.radam import RiemannianAdam
from manifold.poincare import PoincareBall
manifold = PoincareBall()


def distance_matrix(nodes):
    length = len(nodes)
    matrix = torch.zeros(length,length)
    for n_idx in range(len(nodes)):
        matrix[n_idx] = manifold.distance(
            torch.unsqueeze(nodes[n_idx],0), nodes) + 1e-8
    matrix = matrix[torch.triu(torch.ones(length, length), diagonal=1) == 1]
    return matrix**2.


class LitHGCN(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.hyp_gcn = model(manifold, 4, 32, 2).double()
        self.lr = lr
        #self.hyp_gcn = model(manifold, 4, 2).double()

    def training_step(self, batch, batch_idx):
        output = self.hyp_gcn(batch)

        loss_temp=0
        for graph_idx in torch.unique(batch.batch):
            graph_mask = batch.batch == graph_idx

            _input = distance_matrix(output[graph_mask])
            _target = distance_matrix(batch.y[graph_mask])

            loss_temp += F.mse_loss(_input,_target)

        loss_temp /= batch.num_graphs
        self.log('training loss', loss_temp, 
            prog_bar=True, batch_size=batch.num_graphs)

        self.logger.experiment.add_scalar('loss/train', loss_temp,
            self.global_step)

        return loss_temp
    

    def validation_step(self, batch, batch_idx):
        output = self.hyp_gcn(batch)

        loss_temp=0
        for graph_idx in torch.unique(batch.batch):
            graph_mask = batch.batch == graph_idx

            _input = distance_matrix(output[graph_mask])
            _target = distance_matrix(batch.y[graph_mask])
            
            loss_temp += F.mse_loss(_input,_target)

        loss_temp /= batch.num_graphs
        self.log('validation loss', loss_temp, batch_size=batch.num_graphs)


    def test_step(self, batch, batch_idx):
        output = self.hyp_gcn(batch)

        loss_tepm=0
        for graph_idx in torch.unique(batch.batch):
            graph_mask = batch.batch == graph_idx

            _input = distance_matrix(output[graph_mask])
            _target = distance_matrix(batch.y[graph_mask])
            
            loss_temp += F.mse_loss(_input,_target)

        loss_temp /= batch.num_graphs
        self.log('test loss', loss_temp, batch_size=batch.num_graphs)


    def configure_optimizers(self):
        optimizer = RiemannianAdam(self.hyp_gcn.parameters(),
            lr=self.lr, weight_decay=5e-4)
        return optimizer


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('loss/epoch', avg_loss,
            self.current_epoch)


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running training ...')
        return bar
