import numpy as np
import torch
from contextlib import ExitStack
import graphicle as gcl

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

from tqdm import tqdm
from heparchy.read.hdf import HdfReader

class ParticleDataset(Dataset):
    '''Particles shower dataset.'''

    def __init__(self,path,process_name='background'):
        self.__stack = ExitStack()
        self.__file_obj = self.__stack.enter_context(HdfReader(path=path))
        self.__process = self.__file_obj.read_process(name=process_name)
       

    def __len__(self):
        return len(self.__process)


    def __adj_matrix(self,eta,phi):
        _eta = eta - eta[...,None]
        _phi_row = phi - phi[...,None]
        _phi = np.min(
            (_phi_row % (2*np.pi), np.abs(-_phi_row % (2*np.pi))), axis=0)
        return np.sqrt(_eta**2 + _phi**2)


    def __getitem__(self,idx):
        _event = self.__process.read_event(idx)
        graph = gcl.Graphicle.from_numpy(
            edges = _event.edges,
            pmu = _event.pmu,
            pdg = _event.pdg,
            final = _event.mask('final'))

        finals_pmu = graph.pmu[graph.final].data
        finals_hyper = _event.get_custom('hyper_coords')[graph.final.data]

        X = torch.tensor(
            np.transpose([finals_pmu['x'], finals_pmu['y'], finals_pmu['z'],
            finals_pmu['e']]), dtype=torch.float64)

        adj_matrix = self.__adj_matrix(
            graph.pmu.eta[graph.final.data],graph.pmu.phi[graph.final.data]) 
        knn = gcl.matrix.knn_adj(adj_matrix, k=10)
        knn = np.maximum(knn, np.transpose(knn))
        edges = np.array(np.where(knn == True))
        edge_index = torch.tensor(edges,dtype=torch.long)

        y = torch.tensor(finals_hyper,dtype=torch.float64)

        data = Data(x=X, edge_index=edge_index, y=y)
        return data


    def __del__(self):
        self.__stack.close()
