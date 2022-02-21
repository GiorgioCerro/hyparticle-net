import numpy as np
import pandas as pd
from time import time
import graphicle as gcl

from hyperTest import HyperEmbedding
from heparchy.read.hdf import HdfReader

dtype = [('loss','<f8'),('size','<f8'),('epochs','<f8'),
        ('steps','<f8'),('lr','<f8'),('neg','<f8'),('time','<f8')]
grid = np.zeros(54,dtype=dtype)

epochs = [40,50,60]
steps = [10,12,14]
lrs = [0.03,0.04,0.05]
negs = [1,2]

number_of_events = 100
with HdfReader('data/hyper_data.hdf5') as hep_file:
    process = hep_file.read_process(name='background')
    count = 0 
    for epoch in epochs:
        for step in steps:
            for neg in negs:
                for lr in lrs:
                    loss = []
                    size = []
                    times = []
                    for event_idx in range(number_of_events):
                        event = process.read_event(event_idx)
                        graph = gcl.Graphicle.from_numpy(
                            edges = event.edges)
                        graph.adj = gcl.transform.particle_as_node(graph.adj)

                        params_dict = {
                            'dim':2,'max_epochs':epoch,'lr':lr,
                            'n_negative':neg,'context_size':1,'steps':step}
                        t = time()
                        hyper = HyperEmbedding(graph,params_dict)
                        loss.append(hyper.get_embedding(return_loss=True))
                        times.append(time()-t)
                        size.append(len(hyper.embeddings))

                    grid[count] = (round(np.mean(loss),3),np.mean(size),epoch,
                        step,lr,neg,round(np.mean(times),3))
                    count += 1

ten_best = np.argsort(grid['loss'])[:10]
f = open('grid_search.txt','w')
f.write('These are the best ten options (computed over 100 samples)')
f.write('\n')
f.write('Col: Mean loss / mean size / epoch / steps / lr / neg / mean time')
f.write('\n')
for idx,best in enumerate(ten_best):
    f.write(f'Option number: {idx}, grid values: {grid[best]}')
    f.write('\n')
f.close()
