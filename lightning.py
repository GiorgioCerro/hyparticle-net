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
    return matrix[torch.triu(torch.ones(length, length), diagonal=1) == 1]


class LitHGCN(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.hyp_gcn = model(manifold, 4, 64, 64, 2).double()

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

        logs={'train loss': loss_temp}
        batch_dictionary={
            'loss': loss_temp,
            'log': logs
        }
        return batch_dictionary
    

    def validation_step(self, batch, batch_idx):
        output = self.hyp_gcn(batch)

        loss_temp=0
        for graph_idx in torch.unique(batch.batch):
            graph_mask = batch.batch == graph_idx

            _input = distance_matrix(output[graph_mask])
            #_input = _input[torch.triu(torch.ones(length, length),
            #    diagonal=1) == 1]

            _target = distance_matrix(batch.y[graph_mask])
            #_target = _target[torch.triu(torch.ones(length, length),
            #    diagonal=1) == 1]
            
            loss_temp += F.mse_loss(_input,_target)

        loss_temp /= batch.num_graphs
        self.log('validation loss', loss_temp, batch_size=batch.num_graphs)

        logs={'valid loss': loss_temp}
        batch_dictionary={
            'loss': loss_temp,
            'log': logs
        }


    def test_step(self, batch, batch_idx):
        output = self.hyp_gcn(batch)

        loss_tepm=0
        for graph_idx in torch.unique(batch.batch):
            graph_mask = batch.batch == graph_idx

            _input = distance_matrix(output[graph_mask])
            #_input = _input[torch.triu(torch.ones(length, length),
            #    diagonal=1) == 1]

            _target = distance_matrix(batch.y[graph_mask])
            #_target = _target[torch.triu(torch.ones(length, length),
            #    diagonal=1) == 1]
            
            loss_temp += F.mse_loss(_input,_target)

        loss_temp /= batch.num_graphs
        self.log('test loss', loss_temp, batch_size=batch.num_graphs)


    def configure_optimizers(self):
        optimizer = RiemannianAdam(self.hyp_gcn.parameters(),
            lr=0.1, weight_decay=5e-4)
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
