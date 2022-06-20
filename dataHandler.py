import numpy as np
import torch
from contextlib import ExitStack
import glob
import graphicle as gcl

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

from tqdm import tqdm
from heparchy.read.hdf import HdfReader
from manifold.poincare import PoincareBall
manifold = PoincareBall()

def matrix_distance(nodes):
    matrix = torch.zeros(len(nodes), len(nodes))
    for n_idx in range(len(nodes)):
        matrix[n_idx] = manifold.distance(
            torch.unsqueeze(nodes[n_idx],0),nodes) + 1e-8
    return matrix**2


class ParticleDataset(Dataset):
    '''Particles shower dataset.'''

    def __init__(self,path,process_name):
        self.__process_name = process_name
        self.__files = glob.glob(path+'/*.hdf5')
        #print(self.__files)
        self.__ranges = [(-1,-1)]

        stack = ExitStack()
        for file in self.__files:
            file_obj = stack.enter_context(HdfReader(path=file))
            process = file_obj.read_process(name=self.__process_name)
            ini = self.__ranges[-1][1] + 1
            fin = ini + len(process) - 1
            self.__ranges.append((ini,fin))
        stack.close()

        __dtype = [('ini','i4'),('fin','i4')]
        self.__ranges = np.array(self.__ranges[1:],dtype=__dtype)


    def __len__(self):
        return self.__ranges['fin'][-1]


    #def __ranges__(self):
    #    return self.__ranges


    def __adj_matrix(self,eta,phi):
        _eta = eta - eta[...,None]
        _phi = phi - phi[...,None]
        _phi = np.min((_phi % (2*np.pi), np.abs(- _phi % (2*np.pi))), axis=0)
        return np.sqrt(_eta**2 + _phi**2)


    def __matrix_distance(self,nodes):
        matrix = torch.zeros(len(nodes), len(nodes))
        for n_idx in range(len(nodes)):
            matrix[n_idx] = manifold.sqdist(
                torch.unsqueeze(nodes[n_idx],0),nodes) + 1e-8
        return matrix


    def __get_classes(self, nodes, links, graph):
        idx_nodes = [np.where(graph.edges['out'] == pt)[0] for pt in nodes]
        true_values = [[],[]]
        false_values = [[],[]]
        for i in range(len(nodes)):
            parent_1 = graph.edges['in'][idx_nodes[i]]
            for j in range(i+1, len(nodes)):
                parent_2 = graph.edges['in'][idx_nodes[j]]
                if len(set(parent_1) & set(parent_2)) > 0:
                    true_values[0].append(i)
                    true_values[1].append(j)
                else:
                    false_values[0].append(i)
                    false_values[1].append(j)
        
        len_sample = len(true_values[0])
        falses = np.array(false_values)
        neg_sample = np.random.randint(0, len(falses[0]), (1,len_sample))[0]

        false_values = np.array(
            [falses[0][neg_sample],falses[1][neg_sample]])
        false_values = torch.tensor(false_values)
        true_values = torch.tensor(true_values)
        return true_values, false_values


    def __getitem__(self,idx):
        _file_idx = int(np.where(
            np.logical_and(np.less_equal(self.__ranges['ini'],idx),
                            np.greater_equal(self.__ranges['fin'],idx)))[0])

        _event_idx = idx - self.__ranges[_file_idx]['ini']
        with HdfReader(path=self.__files[_file_idx]) as hep_file:
            process = hep_file.read_process(name=self.__process_name)
            _event = process.read_event(_event_idx)

            graph = gcl.Graphicle.from_numpy(
                edges = _event.edges,
                pmu = _event.pmu,
                pdg = _event.pdg,
                final = _event.mask('final'))

            graph.adj = gcl.transform.particle_as_node(graph.adj)
            finals_pmu = graph.pmu[graph.final].data
            finals_nodes = graph.nodes[graph.final]
            finals_hyper = _event.get_custom('hyper_coords')[graph.final.data]

        X = torch.tensor(
            np.transpose([finals_pmu['x'],finals_pmu['y'],finals_pmu['z'],
                finals_pmu['e']]),dtype=torch.float64)

        adj_matrix = self.__adj_matrix(
            graph.pmu.eta[graph.final.data],
                graph.pmu.phi[graph.final.data])
        #_x = manifold.lorentz_to_poincare(X)
        #adj_matrix = np.array(self.__matrix_distance(_x))
        knn = gcl.matrix.knn_adj(adj_matrix, k=3)
        knn = np.maximum(knn, np.transpose(knn))
        edges = np.array(np.where(knn == True))
        edge_index = torch.tensor(edges,dtype=torch.long)

        y = self.__get_classes(finals_nodes, edge_index, graph)
        #y = torch.tensor(classes, dtype=torch.float64)

        data = Data(x=X, edge_index=edge_index, y=y)
        return data
