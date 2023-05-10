import wandb
from torch_geometric.loader import DataLoader

from hyparticlenet.hgnn.train import train
from hyparticlenet.hgnn.train import HGNN_CONFIG
from hyparticlenet.hgnn.util import wandb_cluster_mode

from lundnet.pyg_dataset import DGLGraphDatasetLund
# Modify config
hp = HGNN_CONFIG

hp.logdir = 'logs'
hp.epochs = 10
hp.batch_size = 32
hp.seed = 234
hp.lr = 0.001

hp.num_class = 2
hp.num_layers = 3
hp.in_features = 5
hp.embed_dim = 5
hp.num_centroid = 100

# Jets Data sets
PATH = '/scratch/gc2c20/data/w_tagging/'

train_dataset = DGLGraphDatasetLund(PATH+'/train_bkg/', PATH+'/train_sig/', 
                                        nev=-1, n_sample=1000)
valid_dataset = DGLGraphDatasetLund(PATH+'/valid_bkg/', PATH+'/valid_sig/', 
                                        nev=-1, n_sample=100)


train_loader = DataLoader(dataset=train_dataset, batch_size=hp.batch_size, 
        shuffle=True, num_workers=2)
val_loader = DataLoader(dataset=valid_dataset, batch_size=hp.batch_size, num_workers=2)


wandb_cluster_mode()

# Training the poincare manifold
print('Training the poincare manifold')
hp.manifold = 'lorentz'
hp.loss_embedding = True
hp.best_model_name = 'best_jets_' + hp.manifold
with wandb.init(project='loss_function', entity='office4005', config=dict(hp)):
    train(train_loader, val_loader, args=hp)

# Training the euclidean manifold
hp.manifold = 'lorentz'
hp.loss_embedding = False
hp.best_model_name = 'best_jets_' + hp.manifold
with wandb.init(project='loss_function', entity='office4005', config=dict(hp)):
    train(train_loader, val_loader, args=hp)

# Training 
#hp.manifold = 'poincare'
#hp.embed_dim = 3
#hp.num_layers = 7
#hp.best_model_name = 'best_jets_' + hp.manifold
#with wandb.init(project='loss_function', entity='office4005', config=dict(hp)):
#    train(train_loader, val_loader, args=hp)
