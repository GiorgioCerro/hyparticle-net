# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
from math import ceil
import copy

import click
from showerpipe.generator import PythiaGenerator
from showerpipe.lhe import split, LheData, count_events
from heparchy.write import HdfWriter
from heparchy.data.event import SignalVertex
from tqdm import tqdm

import sys
sys.path.append('/mainfs/home/gc2c20/myproject/hyperparticle/')
sys.path.append('/mainfs/home/gc2c20/myproject/hyperparticle/data_generation/')

import numpy as np
import graphicle as gcl
import fastjet as fj
from lundnet.JetTree import JetTree
from uproot3_methods import TLorentzVectorArray, TLorentzVector

import networkx as nx

@click.command()
@click.argument('lhe_path', type=click.Path(exists=True))
@click.argument('pythia_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(path_type=Path))
@click.argument('process_name',type=click.STRING)
def main(lhe_path, pythia_path, output_filepath,process_name):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank: int = comm.Get_rank()
    num_procs: int = comm.Get_size()

    total_num_events = count_events(lhe_path)
    stride = ceil(total_num_events / num_procs)

    # split filepaths for each process
    split_dir = output_filepath.parent
    split_fname = f'{output_filepath.stem}-{rank}{output_filepath.suffix}'
    split_path = split_dir / split_fname

    if rank == 0:  # split up the lhe file
        lhe_splits = split(lhe_path, stride)
        data = next(lhe_splits)
        for i in range(1, num_procs):
            comm.send(next(lhe_splits), dest=i, tag=10+i)
    else:
        data = comm.recv(source=0, tag=10+rank)

    gen = PythiaGenerator(pythia_path, data)
    if rank == 0:  # progress bar on root process
        gen = tqdm(gen)

    #defining jetdef
    jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.0)
    tree_type = ['akt', 'CA', 'kt']
    jet_def_ls = [
        fj.JetDefinition(fj.antikt_algorithm, 1000.0),
        fj.JetDefinition(fj.cambridge_algorithm, 1000.0),
        fj.JetDefinition(fj.kt_algorithm, 1000.0)
    ]
    with HdfWriter(split_path) as hep_file:
        with hep_file.new_process(process_name) as proc:
            stop_condition = 0 
            for event in gen:
                graph = gcl.Graphicle.from_event(event)
                # cluster
                
                jet_masks = gcl.select.fastjet_clusters(
                    graph.pmu, radius=1., p_val=-1, top_k=2)

                for mask in jet_masks:
                    pt_max = max(graph.pmu.pt[mask])
                    if pt_max >= 500 and pt_max <= 550 and sum(mask) > 5:
                        pmu_jet = graph.pmu.data[mask]
                        constits = [fj.PseudoJet(_pmu[0], _pmu[1], _pmu[2],
                                _pmu[3]) for _pmu in graph.pmu.data[mask][graph.final.data[mask]]]

                        jet = jet_def(constits)[0]
                        if len(jet.constituents()) > 2:
                            data = _build_tree(JetTree(jet))
                            tree_pmu = np.array(list(
                                nx.get_node_attributes(data, 'coordinates').values()))
                            #tree_lund = np.array(list(
                            #    nx.get_node_attributes(data, 'features').values()))

                            if graph.pmu.data.shape[0] > 1 and tree_pmu.shape[0] > 1 and np.array(data.edges).shape[0] > 1:
                                with proc.new_event() as event_write:
                            
                                    event_write.pmu = graph.pmu.data[mask]
                                    event_write.pdg = graph.pdg.data[mask]
                                    event_write.status = graph.status.data[mask]
                                    event_write.edges = graph.edges[mask]
                                    event_write.masks['final'] = graph.final.data[mask]
                                  
                                    for k in range(3):
                                        jt = jet_def_ls[k](constits)[0]
                                        data = _build_tree(JetTree(jet))
                                        tree_pmu = np.array(list(
                                            nx.get_node_attributes(data, 'coordinates').values()))
 
                                        event_write.custom['tree_pmu' + tree_type[k]] = tree_pmu
                                        event_write.custom['tree_edges' + tree_type[k]] = np.array(data.edges)
     
                


def _build_tree(root):
    g = nx.Graph()
    jet_p4 = TLorentzVector(*root.node)

    def _rec_build(nid, node):
        branches = [node.harder, node.softer] 
        for branch in branches:
            if branch is None:# or branch.lundCoord is None:
                # stop when reaching the leaf nodes
                # we do not add the leaf nodes to the graph/tree as they do no have Lund coords
                continue
            cid = g.number_of_nodes()
            node_p4 = TLorentzVector(*branch.node)
            spatialCoord = np.array([node_p4.x, node_p4.y, node_p4.z, node_p4.E])
            g.add_node(cid, coordinates=spatialCoord)#, features=branch.lundCoord.state())
            g.add_edge(cid, nid)
            _rec_build(cid, branch)

    # add root
    #g.add_node(0, coordinates=np.zeros(4, dtype='float32'))
    g.add_node(0, coordinates=np.array([jet_p4.x, jet_p4.y, jet_p4.z, jet_p4.E]))
        #features=root.lundCoord.state())
    _rec_build(0, root)

    return g


if __name__ == '__main__':
    sys.exit(main())
