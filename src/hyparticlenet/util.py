import torch
import dgl
import numpy as np

import operator as op
from pathlib import Path
from typing import List

import heparchy
import click

from tqdm import tqdm


def dot(a, b):
    return torch.bmm(a.unsqueeze(-2), b.unsqueeze(-1)).squeeze(-1)

def atanh(x, EPS):
	values = torch.min(x, torch.Tensor([1.0 - EPS]).to(x.device))
	return 0.5 * (torch.log(1 + values + EPS) - torch.log(1 - values + EPS))

def clamp_min(x, min_value):
	t = torch.clamp(min_value - x.detach(), min=0)
	return x + t

def wandb_cluster_mode():
    """
    Get wandb key and turn wandb offline. Requires os imported?
    """
    import os
    key = os.environ.get("WANDB_KEY")
    os.environ['WANDB_API_KEY'] = key 
    os.environ['WANDB_MODE'] = 'offline'
    #os.environ['WANDB_MODE'] = 'online'

def collate_fn(batch):
    graphs, targets = zip(*batch)
    return dgl.batch(graphs), torch.hstack(targets)


def count_params(model: torch.nn.Module) -> int:
    param_flats = map(op.methodcaller("view", -1), model.parameters())
    param_shapes = map(op.attrgetter("shape"), param_flats)
    param_lens = map(op.itemgetter(0), param_shapes)
    return sum(param_lens)

def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._open_file()

def ROC_area(signal_eff, background_eff):
      """Area under the ROC curve.
      """
      normal_order = signal_eff.argsort()
      return torch.trapz(background_eff[normal_order], signal_eff[normal_order]).item()


def bkg_rejection_at_threshold(signal_eff, background_eff, sig_eff=0.5):
    """Background rejection at a given signal efficiency."""
    return 1 / (1 - background_eff[torch.argmin(torch.abs(signal_eff - sig_eff)) + 1])


EX = {".h5", ".hdf5"}
def evt_read(split_dir: Path):
    split_dir = Path(split_dir)
    fnames = filter(lambda f: f.suffix in EX, split_dir.iterdir())
    fnames_ord: List[Path] = sorted(list(fnames))
    for i, fpath in enumerate(fnames_ord):
        click.echo(f"Copying file {i} of {len(fnames_ord)}")
        with heparchy.read.hdf.HdfReader(fpath) as hep_file:
            try:
                for event in tqdm(hep_file["signal"]):
                    yield event
            except KeyError:
                for event in tqdm(hep_file['background']):
                    yield event


def merge_splits(full: bool, include_custom: bool, split_dir: Path,
        output: Path, process_name: str = "default") -> None:
    with heparchy.write.hdf.HdfWriter(output, compression_level=9) as hep_out:
        with hep_out.new_process(process_name) as proc:
            for event_out, event_in in tqdm(proc.event_iter(evt_read(split_dir))):
                if include_custom:
                    for name, custom in event_in.custom.items():
                        event_out.custom[name] = custom
                final = event_in.masks["final"]
                if full is False:
                    event_out.pmu = event_in.pmu[final]
                    event_out.pdg = event_in.pdg[final]
                    for name, mask in event_in.masks.items():
                        event_out.masks[name] = mask[final]
                    continue
                event_out.pmu = event_in.pmu
                event_out.pdg = event_in.pdg
                event_out.status = event_in.status  # type: ignore
                event_out.edges = event_in.edges
                event_out.masks['final'] = final
                #for name, mask in event_in.masks.items():
                #    event_out.masks[name] = mask
