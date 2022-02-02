import numpy as np
import pandas as pd
import networkx as nx
import random

def random_walk_numpy(graph):
    steps = 10

    df = pd.DataFrame(graph.edges)
    fd = df.rename(columns={'in':'out','out':'in'})
    df = pd.concat([df,fd]) #this is for generating undirected graph

    edge_pivot = df.pivot_table(index='in',values='out',aggfunc=lambda x: x.tolist())
    table = np.zeros((len(edge_pivot),steps+1))
    table[:,0] = edge_pivot.index.values

    for i in range(steps):
        ls = edge_pivot.loc[table[:,i]].values
        table[:,i+1] = np.apply_along_axis(lambda x: np.random.choice(x[0]),1,ls)

    return table


def _walk(G,node,steps):
    path_temp = [node]
    for i in range(steps):
        current_step = path_temp[-1]
        next_step = random.choice(list(G.neighbors(current_step)))
        path_temp.append(next_step)

    return np.array(path_temp)

def random_walk_list(graph):
    steps=10
    edge_dict = graph.adj.to_dicts()
    G = nx.Graph(edge_dict['edges'])
    path = []
    for node in G.nodes:
        path.append(_walk(G,node,steps))

    return np.array(path)
