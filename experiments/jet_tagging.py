import sys
import torch
import wandb
import time
import contextlib as ctx
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import click

from hyparticlenet.hgnn.util import wandb_cluster_mode, count_params, collate_fn
from hyparticlenet.hgnn.util import worker_init_fn, ROC_area
from hyparticlenet.hgnn.models.graph_classification import GraphClassification
from hyparticlenet.hgnn.nn.manifold import EuclideanManifold, PoincareBallManifold, LorentzManifold

from dgl.dataloading import GraphDataLoader
from hyparticlenet.data_handler import ParticleDataset

from torchmetrics import MetricCollection, ROC, classification as metrics

NUM_GPUS = torch.cuda.device_count()
#NUM_THREADS = 4
#torch.set_num_threads = NUM_THREADS

import warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")



def training_loop(rank, device, model, optim, dataloader):
    loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
    soft = torch.nn.Softmax(dim=1)
    metric_scores = MetricCollection(dict(
          accuracy = metrics.BinaryAccuracy(),
          precision = metrics.BinaryPrecision(),
          recall = metrics.BinaryRecall(),
          f1 = metrics.BinaryF1Score(),
    )).to(device)
    total_loss = 0

    model.train()
    for graph, label in dataloader:
        label = label.to(device).squeeze().long()
        num_graphs = label.shape[0]

        optim.zero_grad()
        logits = model(graph.to(device))

        loss = loss_function(logits, label)

        pred = soft(logits)[:, 1]
        metric_scores.update(pred, label)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1 - 1e-5)

        total_loss += loss.item() * num_graphs
        optim.step()

    scores = metric_scores.compute()
    if rank == 0:
        print(
            f"loss: {(total_loss/len(dataloader)):.5f}, "
            f"accuracy: {scores['accuracy'].item():.1%}, "
            f"precision: {scores['precision'].item():.1%}, "
            f"recall: {scores['recall'].item():.1%}, "
            f"f1: {scores['f1'].item():.1%}, "
        )

        # Log to wandb
        #wandb.log({
        #    'validation_auc': val_auc,
        #    'validation_accuracy': val_acc,
        #    'validation_loss': val_loss,
        #    'accuracy': scores['accuracy'].item(),
        #    'precision': scores['precision'].item(),
        #    'recall': scores['recall'].item(),
        #    'f1': scores['f1'].item(),
        #    'loss': total_loss/len(dataloader),
        #})

    metric_scores.reset()
    return model, optim


def evaluate(rank, device, model, dataloader):
    loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
    soft = torch.nn.Softmax(dim=1)
    metric_scores = MetricCollection(dict(
          accuracy = metrics.BinaryAccuracy(),
          ROC = ROC(task="binary"),
    )).to(device)
    loss_temp = 0

    model.eval()
    with torch.no_grad():
        for graph, label in dataloader:
            label = label.to(device).squeeze().long()
            num_graphs = label.shape[0]

            logits = model(graph.to(device))
            pred = soft(logits)[:, 1]
            metric_scores.update(pred, label)

            loss_temp += loss_function(logits, label).item() * num_graphs
    
    scores = metric_scores.compute()
    accuracy = scores["accuracy"].item()
    fpr, tpr, threshs = scores["ROC"]
    eff_s = tpr
    eff_b = 1 - fpr
    auc = ROC_area(eff_s, eff_b)
        
    if rank==0:
        print(
                f"validation: "
                f"accuracy: {accuracy:.1%}, "
                f"auc: {auc:.1%}, "
                f"loss: {loss_temp / len(dataloader):.5f}, "
        ) 
    return 


@ctx.contextmanager
def enter_process_group(world_size: int, rank: int):
    """Context manager to provide a DDP process group for Pytorch's
    multiprocessing spawner.
    """
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12355',
        world_size=world_size,
        rank=rank,
    )
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    try:
        yield device
    finally:
        torch.distributed.destroy_process_group()


def train(rank, args, dataset, valid_dataset):
    with enter_process_group(NUM_GPUS, rank) as device:
        if args.manifold == 'euclidean':
            manifold = EuclideanManifold()
        elif args.manifold == 'poincare':
            manifold = PoincareBallManifold()
        elif args.manifold == 'lorentz':
            manifold = LorentzManifold()
        else:
            manifold = EuclideanManifold()
            warnings.warn('No valid manifold was given as input, using Euclidean as default')

        model = GraphClassification(args, manifold).to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
        lr_steps = [20, 30]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, 
            milestones=lr_steps, gamma=0.1)

        if rank == 0:
            print(f"Model with {count_params(model)} trainable parameters")
            print(f"Training over {len(dataset)} events")
        #print(f"Validation dataset contains {len(valid_dataset)} events")
        for epoch in range(args.epochs):
            if rank == 0:
                print(f'Epoch: {epoch:n}')
                if epoch == args.epochs-1:
                    print(f'Final epoch')
            init = time.time()
            dataloader = GraphDataLoader(
                dataset=dataset, 
                batch_size=args.batch_size, 
                num_workers=20,
                drop_last=True,
                shuffle=True, 
                use_ddp=True,
                pin_memory=True,
                pin_memory_device=str(device),
                prefetch_factor=16,
                collate_fn=collate_fn,
                worker_init_fn=worker_init_fn,
            )

            val_loader = GraphDataLoader(
                dataset=valid_dataset, 
                batch_size=args.batch_size,
                num_workers=20,
                drop_last=True,
                use_ddp=True,
                pin_memory=True,
                pin_memory_device=str(device),
                prefetch_factor=16,
                collate_fn=collate_fn,
                worker_init_fn=worker_init_fn,
            )
            #    collate_fn=collate_fn)

            model, optim = training_loop(rank, device, model, optim, dataloader)
            evaluate(rank, device, model, val_loader)
            #scheduler.step()
            
            if rank == 0: 
                print(f'epoch time: {(time.time() - init):.2f}')
                print(10*'~')
                p = Path(args.logdir)
                p.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), p.joinpath(f'{args.best_model_name}.pt'))


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):
    #config_path = 'configs/jets_config.yaml'
    args = OmegaConf.load(config_path)
    print(f"Working with the following configs:")
    for key, val in args.items():
        print(f"{key}: {val}")

    # Jets Data sets
    PATH = '/scratch/gc2c20/data/jet_tagging'
    train_dataset = ParticleDataset(Path(PATH + '/train_sig.hdf5'), 
                Path(PATH + '/train_bkg.hdf5'), num_samples=args.train_samples)
    valid_dataset = ParticleDataset(Path(PATH + '/valid_sig.hdf5'), 
                Path(PATH + '/valid_bkg.hdf5'), num_samples=args.valid_samples)
    

    args.best_model_name = 'best_' + args.manifold + '_dim' + str(args.embed_dim)
    torch.multiprocessing.spawn(train, args=(args, train_dataset, valid_dataset,), nprocs=NUM_GPUS)


if __name__=='__main__':
    sys.exit(main())
