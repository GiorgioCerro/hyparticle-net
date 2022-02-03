import numpy as np
import pandas as pd
from random import choice
import geomstats 
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_ball import PoincareBall

class HyperEmbedding:
    def __init__(self,graph,params_dict={
                        'dim':2,
                        'max_epochs':50,
                        'lr':.05,
                        'n_negative':2,
                        'context_size':1,
                        'steps':10}):

        self.dim = params_dict['dim']
        self.max_epoch = params_dict['max_epochs']
        self.lr = params_dict['lr']
        self.n_negative = params_dict['n_negative']
        self.context_size = params_dict['context_size']
        self.steps = params_dict['steps']

        df = pd.DataFrame(graph.edges)
        self.df = pd.concat([df,df.rename(columns={'in':'out','out':'in'})])
        self.edge_pivot = self.df.pivot_table(index='in',values='out',aggfunc=lambda x: x.tolist())
        
        self.size = len(self.edge_pivot)


    def random_walk(self):
        record = np.empty((self.steps+1,self.size),dtype='<i4')
        record[0] = self.edge_pivot.index.values
        for step in range(self.steps):
            record[step+1,:] = self.edge_pivot.loc[(record[step,:],'out')].apply(choice)

        return record



