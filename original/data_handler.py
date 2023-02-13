import numpy as np
from contextlib import ExitStack
import glob
import graphicle as gcl
import torch
import networkx as nx

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

from tqdm import tqdm
from heparchy.read.hdf import HdfReader

from manifold.poincare import PoincareBall
class ParticleDataset(Dataset):
    '''Particles shower dataset.'''

    def __init__(self, path) -> None:
        self.__files = glob.glob(path + '/*.hdf5')
        self.__ranges = [(-1, -1)]

        stack = ExitStack()
        for file in self.__files:
            file_obj = stack.enter_context(HdfReader(path=file))
            #process = file_obj.read_process(name = self.__process_name)
            try: 
                process = file_obj.read_process(name = 'signal')
                self.__process_name = 'signal'
                self.label = 1
            except KeyError: 
                process = file_obj.read_process('background')
                self.__process_name = 'background'
                self.label = 0
            ini = self.__ranges[-1][1] + 1
            fin = ini + len(process) - 1
            self.__ranges.append((ini, fin))
        stack.close()

        _dtype = [('ini', 'i4'), ('fin', 'i4')]
        self.__ranges = np.array(self.__ranges[1:], dtype=_dtype)
        
        self.algo = ['aKt', 'CA', 'Kt']
        self.manifold = PoincareBall()

    def __len__(self):
        return self.__ranges['fin'][-1]


    def __getitem__(self, idx):
        _file_idx = int(np.where(
            np.logical_and(np.less_equal(self.__ranges['ini'], idx),
                            np.greater_equal(self.__ranges['fin'], idx)))[0])

        _event_idx = idx - self.__ranges[_file_idx]['ini']
        event_dict = {}
        with HdfReader(path=self.__files[_file_idx]) as hep_file:
            process = hep_file.read_process(name=self.__process_name)
            _event = process.read_event(_event_idx)

            k = 0
            mask = _event.get_custom(self.algo[k] + '_mask')
            pmu = _event.get_custom(self.algo[k] + '_pmu')[mask]
            hyp = _event.get_custom(self.algo[k] + '_hyp')[mask]
            edges = _event.get_custom(self.algo[k] + '_edges')[mask][:-1]
            #edges = edges.view((edges.dtype[0], len(edges.dtype.names)))
       
        G = nx.Graph()
        G.add_edges_from(edges)
        nodes = np.array(G.nodes())
        mapping = {nodes[i]: i for i in range(len(nodes))}
        G = nx.relabel_nodes(G, mapping)
        edges = torch.tensor(list(G.edges))
     
        X = torch.tensor(
                np.transpose([pmu['x'],pmu['y'],pmu['z'],pmu['e'] ]), 
                dtype=torch.float64
        )
        y = torch.tensor(self.label)

        
        data = Data(
            x = X, 
            edge_index = edges.t().contiguous(), 
            y = y,
        )
        return data