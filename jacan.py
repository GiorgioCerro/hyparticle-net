import numpy as np
import pandas as pd
import networkx as nx
import random
import graphicle
from heparchy.read.hdf import HdfReader
with HdfReader('data/hyper_data.hdf5') as hep_file:
    process = hep_file.read_process(name='background')
    event = process.read_event(0)
    graph = graphicle.Graphicle.from_numpy(
            edges=event.edges
    )

steps = 10

df = pd.DataFrame(graph.edges)
fd = df.rename(columns={'in':'out','out':'in'})
df = pd.concat([df,fd]) #this is for generating undirected graph

edge_pivot = df.pivot_table(index='in',values='out',aggfunc=lambda x: x.tolist())
table = np.zeros((len(edge_pivot),steps+1))
table[:,0] = edge_pivot.index.values

edge_dict = graph.adj.to_dicts()
G = nx.Graph(edge_dict['edges'])

def random_walk_numpy(graph):
    table = np.zeros((len(edge_pivot),steps+1))
    table[:,0] = edge_pivot.index.values
    vfunc = np.vectorize(lambda x: random.choice(x))
    for i in range(steps):
        ls = edge_pivot.loc[table[:,i]].values
        table[:,i+1] = np.reshape(vfunc(ls),len(table))

    return table


def _walk(G,node,steps):
    path_temp = [node]
    for i in range(steps):
        current_step = path_temp[-1]
        next_step = random.choice(list(G.neighbors(current_step)))
        path_temp.append(next_step)

    return np.array(path_temp)

def random_walk_list(graph):
    #edge_dict = graph.adj.to_dicts()
    #G = nx.Graph(edge_dict['edges'])
    path = []
    for node in G.nodes:
        path.append(_walk(G,node,steps))

    return np.array(path)


import graphicle
from heparchy.read.hdf import HdfReader
with HdfReader('data/hyper_data.hdf5') as hep_file:
    process = hep_file.read_process(name='background')
    event = process.read_event(0)
    shower = graphicle.Graphicle.from_numpy(
            edges=event.edges
    )

#def main():
#    #FormJets.SpectralFull(ew,assign=True,**spectral_jet_params)
#    random_walk_numpy(shower)
#if __name__ == '__main__':
#    import cProfile
#    cProfile.run('main()','output.dat')
#
#    import pstats
#    from pstats import SortKey
#
#    with open('output_time.txt','w') as f:
#        p = pstats.Stats('output.dat',stream=f)
#        p.sort_stats('time').print_stats()
#
#    with open('output_calls.txt','w') as f:
#        p = pstats.Stats('output.dat',stream=f)
#        p.sort_stats('calls').print_stats()
