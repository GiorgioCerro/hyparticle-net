import wandb
from torch_geometric.loader import DataLoader

from hyparticlenet.hgnn.train import train
from hyparticlenet.hgnn.util import wandb_cluster_mode

from lundnet.pyg_dataset import DGLGraphDatasetLund

from omegaconf import OmegaConf
config_path = 'configs/jets_config.yaml'
args = OmegaConf.load(config_path)
args.manifold = 'lorentz'
args.batch_size = 64
args.embed_dim = 20
args.train_samples = 50_000
args.valid_samples = 5_000

# Jets Data sets
PATH = '/scratch/gc2c20/data/w_tagging/'

train_dataset = DGLGraphDatasetLund(PATH+'/train_bkg/', PATH+'/train_sig/', 
                                nev=-1, n_sample=args.train_samples)
valid_dataset = DGLGraphDatasetLund(PATH+'/valid_bkg/', PATH+'/valid_sig/', 
                                nev=-1, n_sample=args.valid_samples)


train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
        shuffle=True)
val_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)


wandb_cluster_mode()

args.loss_embedding = False
args.best_model_name = 'best_jets_' + args.manifold
with wandb.init(project='loss_function', entity='office4005', config=dict(args)):
    train(train_loader, val_loader, args=args)



