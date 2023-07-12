import sys
import torch
import wandb
import time
import contextlib as ctx
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import click
from collections import OrderedDict

from hyparticlenet.util import wandb_cluster_mode, count_params, collate_fn
from hyparticlenet.util import worker_init_fn, ROC_area, bkg_rejection_at_threshold
from hyparticlenet.hgnn import HyperbolicGNN

from lundnet.LundNet import LundNet

from dgl.dataloading import GraphDataLoader
from hyparticlenet.data_handler import ParticleDataset

from torchmetrics import MetricCollection, ROC, classification as metrics

NUM_GPUS = torch.cuda.device_count()
#NUM_THREADS = 4
#torch.set_num_threads = NUM_THREADS

import warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")


def training_loop(rank, args, device, model, optim, scheduler, dataloader, val_loader):
    loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
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
        loss.backward()
        optim.step()

        pred = soft(logits)[:, 1]
        metric_scores.update(pred, label)
        total_loss += loss.item()

        if args.manifold == "poincare":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1 - 1e-5)


    scores = metric_scores.compute()
    # evaluation
    scores_eval, val_loss = evaluate(device, model, val_loader)
    fpr, tpr, threshs = scores_eval["ROC"]
    eff_s = tpr
    eff_b = 1 - fpr
    auc = ROC_area(eff_s, eff_b)
    bkg_rej_05 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.5)
    bkg_rej_07 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.7)
        
    if rank == 0:
        print(
            f"loss: {(total_loss/len(dataloader)):.5f}, "
            f"accuracy: {scores['accuracy'].item():.1%}, "
            f"precision: {scores['precision'].item():.1%}, "
            f"recall: {scores['recall'].item():.1%}, "
            f"f1: {scores['f1'].item():.1%}, "
            f"\n validation: "
            f"val_acc: {scores_eval['accuracy'].item():.1%}, "
            f"val_auc: {auc:.1%}, "
            f"val_loss: {val_loss:.5f}, "
        )

    wandb.log({
        "accuracy": scores['accuracy'].item(),
        "precision": scores['precision'].item(),
        "recall": scores['recall'].item(),
        "f1": scores['f1'].item(),
        "loss": total_loss/len(dataloader),
        "val/accuracy": scores_eval['accuracy'].item(),
        "val/auc": auc,
        "val/loss": val_loss,
        "val/bkg_rej_05": bkg_rej_05,
        "val/bkg_rej_07": bkg_rej_07,
    })

    #scheduler.step(val_loss)
    scheduler.step()
    metric_scores.reset()
    return model, optim


def evaluate(device, model, dataloader):
    loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
    soft = torch.nn.Softmax(dim=1)
    metric_scores_eval = MetricCollection(dict(
          accuracy = metrics.BinaryAccuracy(),
          precision = metrics.BinaryPrecision(),
          recall = metrics.BinaryRecall(),
          f1 = metrics.BinaryF1Score(),
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
            metric_scores_eval.update(pred, label)

            loss_temp += loss_function(logits, label).item() #* num_graphs
    
    scores_eval = metric_scores_eval.compute()
    #metric_scores_eval.reset()
    return scores_eval, loss_temp/len(dataloader)


@ctx.contextmanager
def enter_process_group(world_size: int, rank: int):
    """Context manager to provide a DDP process group for Pytorch's
    multiprocessing spawner.
    """
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12359",
        world_size=world_size,
        rank=rank,
    )
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    try:
        yield device
    finally:
        torch.distributed.destroy_process_group()


def train(rank, args, dataset, valid_dataset, test_dataset, group_id):
    with ctx.ExitStack() as stack:
        device = stack.enter_context(enter_process_group(NUM_GPUS, rank))
        _ = stack.enter_context(
                wandb.init(project="HyperGNN-edgeconv", entity="office4005", config=dict(args),
                group=group_id)
        )
        if args.model == "lundnet5":
            conv_params = [[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]]
            fc_params = [(256, 0.1)]
            use_fusion = True
            batch_size = args.batch_size
            model = LundNet(input_dims=5, num_classes=2, fc_params=fc_params,
                            use_fusion=use_fusion).to(device)
        else:
            model = HyperbolicGNN(input_dims=5, num_centroid=args.num_centroid, 
                            num_classes=2, manifold=args.manifold).to(device)
        #model = GraphClassification(args, manifold)
        #state_dict = torch.load("logs/" + args.best_model_name + ".pt", map_location="cpu")
        #dt = OrderedDict()
        #for key, val in state_dict.items():
        #    dt[key[7:]] = val
        #model.load_state_dict(dt)
        #model.to(device)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device, find_unused_parameters=True)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", 
        #    patience=10, factor=0.5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10, 20],
                                                        gamma=0.1)
        if rank == 0:
            print(f"Model with {count_params(model)} trainable parameters")
            print(f"Training over {len(dataset)} events")
        #print(f"Validation dataset contains {len(valid_dataset)} events")
        for epoch in range(args.epochs):
            if rank == 0:
                print(f"Epoch: {epoch:n}")
                if epoch == args.epochs-1:
                    print(f"Final epoch")
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

            model, optim = training_loop(rank, args, device, model, optim, 
                                        scheduler, dataloader, val_loader)
            
            if rank == 0: 
                print(f"epoch time: {(time.time() - init):.2f}")
                print(10*"~")
                p = Path(args.logdir)
                p.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), p.joinpath(f"{args.best_model_name}.pt"))

        
        print(20*"=")
        print(f"Training complete")
        print(f"Testing on {len(test_dataset)} events.")
        test_loader = GraphDataLoader(
            dataset=test_dataset,
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
        scores_test, test_loss = evaluate(device, model, test_loader)
        fpr, tpr, threshs = scores_test["ROC"]
        eff_s = tpr
        eff_b = 1 - fpr
        auc = ROC_area(eff_s, eff_b)
        bkg_rej_05 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.5)
        bkg_rej_07 = bkg_rejection_at_threshold(eff_s, eff_b, sig_eff=0.7)
        
        print(f"Accuracy: {scores_test['accuracy'].item():.5f}")
        print(f"AUC: {auc:.5f}")
        print(f"Loss: {test_loss:.5f}")
        print(f"Inv_bkg_at_sig_05: {bkg_rej_05:.5f}")
        print(f"Inv_bkg_at_sig_07: {bkg_rej_07:.5f}")


@click.command()
@click.option("--model", type=click.Choice(["hgnn-euclidean", "hgnn-poincare",
            "hgnn-lorentz", "lundnet5"]), default="hgnn-euclidean")
@click.argument("num_classes", type=click.INT, default=2)
@click.argument("num_centroid", type=click.INT, default=100)
@click.argument("lr", type=click.FLOAT, default=0.001)
@click.argument("dropout", type=click.FLOAT, default=0.)
@click.argument("batch_size", type=click.INT, default=128)
@click.argument("epochs", type=click.INT, default=30)
@click.argument("data_path", type=click.Path(exists=True), 
            default="/scratch/gc2c20/data/jet_tagging")
@click.argument("train_samples", type=click.INT, default=1_000_000)
@click.argument("valid_samples", type=click.INT, default=100_000)
@click.argument("test_samples", type=click.INT, default=100_000)
def main(**kwargs):
    args = OmegaConf.create(kwargs)
    if args.model != "lundnet5":
        args.manifold = args.model[5:]
    else:
        args.manifold = "none"
    print(f"Working with the following configs:")
    for key, val in args.items():
        print(f"{key}: {val}")

    # Jets Data sets
    PATH = args.data_path
    train_dataset = ParticleDataset(Path(PATH + "/train_sig.hdf5"), 
                Path(PATH + "/train_bkg.hdf5"), num_samples=args.train_samples)
    valid_dataset = ParticleDataset(Path(PATH + "/valid_sig.hdf5"), 
                Path(PATH + "/valid_bkg.hdf5"), num_samples=args.valid_samples)
    test_dataset = ParticleDataset(Path(PATH + "/test_sig.hdf5"),
                Path(PATH + "/test_bkg.hdf5"), num_samples=args.valid_samples)
    

    args.logdir = "logs/"
    args.best_model_name = "best_" + args.model

    wandb_cluster_mode()
    group_id = args.model + "-2"
    torch.multiprocessing.spawn(train, args=(args, train_dataset, valid_dataset,
        test_dataset, group_id), nprocs=NUM_GPUS)



if __name__=="__main__":
    sys.exit(main())
