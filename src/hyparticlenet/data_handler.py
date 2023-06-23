import numpy as np
from contextlib import ExitStack
import graphicle as gcl
import torch
import dgl
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from heparchy.read.hdf import HdfReader

#from manifold.poincare import PoincareBall
class ParticleDataset(Dataset):
    '''Particles shower dataset.'''

    def __init__(
        self, 
        sig_path: Path, 
        bkg_path: Path,
        num_samples: int =-1, 
        open_at_init: bool = False,
    ) -> None:
        self.sig_path = sig_path
        self.bkg_path = bkg_path
        self._stack = ExitStack()
        self.num_samples = num_samples
        #self.indices = range(int(self.num_samples / 2.))
        self.indices = range(int(self.__len__() / 2.))
        if open_at_init is True:
            self._open_file()
       

    def __len__(self) -> int:
        with HdfReader(self.sig_path) as hep_file:
            sig_size = len(hep_file["signal"])
        with HdfReader(self.bkg_path) as hep_file:
            bkg_size = len(hep_file["background"])

        return min(self.num_samples, min(sig_size, bkg_size) * 2)


    def __getitem__(self, idx: int) -> tuple[dgl.DGLGraph, torch.Tensor]:
        half_samples = int(self.__len__() / 2.)
        if idx < half_samples:
            event = self._sig_events[self.indices[idx]]
            label = torch.tensor(0)

        else:
            event = self._bkg_events[self.indices[idx - half_samples]]
            label = torch.tensor(1)

        graph = self._generate_graph(event)
        return (graph, label)


    def _generate_graph(self, event):
        coordinates = torch.tensor(event.custom['tree_pmu']).to(torch.float32)
        features = torch.tensor(event.custom['tree_lund']).to(torch.float32)
        edges = torch.tensor(event.custom['tree_edges'])
        edges = torch.cat((edges, edges.flip(1)), dim=0)

        graph = dgl.graph((edges[:,0], edges[:,1]))
        graph.ndata['coordinates'] = coordinates
        graph.ndata['features'] = features
        pmu = gcl.MomentumArray(coordinates)
        graph.ndata['pmu'] = torch.tensor(np.array([
            pmu.eta, pmu.phi, pmu.pt, pmu.mass])).transpose(0, 1).to(torch.float32)

        return graph


    def _open_file(self) -> None:
        self._stack = ExitStack()
        sig_file = self._stack.enter_context(HdfReader(self.sig_path))
        bkg_file = self._stack.enter_context(HdfReader(self.bkg_path))
        
        self._sig_events = sig_file["signal"]
        self._bkg_events = bkg_file["background"]

       

"""
class ParticleDataset(Dataset):
    '''Particles shower dataset.'''

    def __init__(self, path) -> None:
        self.__files = glob.glob(path + '/*.hdf5')
        self.__ranges = [(-1, -1)]

        stack = ExitStack()
        for file in self.__files:
            file_obj = stack.enter_context(HdfReader(path=file))
            try: 
                process = file_obj['signal']
                self.__process_name = 'signal'
                self.label = 1
            except KeyError: 
                process = file_obj['background']
                self.__process_name = 'background'
                self.label = 0
            ini = self.__ranges[-1][1] + 1
            fin = ini + len(process) - 1
            self.__ranges.append((ini, fin))
        stack.close()

        _dtype = [('ini', 'i4'), ('fin', 'i4')]
        self.__ranges = np.array(self.__ranges[1:], dtype=_dtype)
        

    def __len__(self):
        return self.__ranges['fin'][-1]


    def __getitem__(self, idx):
        _file_idx = int(np.where(
            np.logical_and(np.less_equal(self.__ranges['ini'], idx),
                            np.greater_equal(self.__ranges['fin'], idx)))[0])

        _event_idx = idx - self.__ranges[_file_idx]['ini']
        with HdfReader(path=self.__files[_file_idx]) as hep_file:
            process = hep_file[self.__process_name]
            _event = process[_event_idx]

            k = 0
            mask = _event.masks["final"]
            pmu = _event.pmu
            edges = _event.edges
       

        G = nx.Graph()
        G.add_edges_from(edges)
        nodes = np.array(G.nodes())
        mapping = {nodes[i]: i for i in range(len(nodes))}
        G = nx.relabel_nodes(G, mapping)
        edges = torch.tensor(list(G.edges))
     
        X = torch.tensor(
                np.transpose([pmu['x'],pmu['y'],pmu['z'],pmu['e'] ]), 
                dtype=torch.float32
        )
        y = torch.tensor(self.label)

        
        data = Data(
            x = X, 
            edge_index = edges.t().contiguous(), 
            y = y,
        )
        return data
"""
