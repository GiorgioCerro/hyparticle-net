import numpy as np
import pandas as pd
from random import choice
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_ball import PoincareBall
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,ConnectionPatch
plt.style.use('dark_background')

class HyperEmbedding:
    def __init__(self,graph,params_dict={
                        'dim':2,
                        'max_epochs':50,
                        'lr':.05,
                        'n_negative':2,
                        'context_size':1,
                        'steps':10}):

        self.dim = params_dict['dim']
        self.max_epochs = params_dict['max_epochs']
        self.lr = params_dict['lr']
        self.n_negative = params_dict['n_negative']
        self.context_size = params_dict['context_size']
        self.steps = params_dict['steps']

        self.graph = graph
        self.df = self._to_dataframe()
        self.edge_pivot = self._to_edge_pivot()
        self.size = len(self.edge_pivot)

        self.embeddings = .2 * np.random.normal(size = (self.size,self.dim))
        self.embeddings.dtype = 'double'
        self.manifold = PoincareBall(2)
 

    def _to_dataframe(self):
        df = pd.DataFrame(self.graph.edges)
        return  pd.concat([df,df.rename(columns={'in':'out','out':'in'})])

    def _to_edge_pivot(self):
        return self.df.pivot_table(
                index='in',values='out',aggfunc=lambda x: x.tolist())


    def random_walk(self):
        '''Generate a random walk for the whole graph
        A node can hop only to its neighbors.

        Returns
        -------
        record : 2d-array
            array.shape = (steps, number of nodes)
        '''
        record = np.empty((self.steps+1,self.size),dtype='<i4')
        record[0] = self.edge_pivot.index.values
        for step in range(self.steps):
            record[step+1,:] = self.edge_pivot.loc[
                    (record[step,:],'out')].apply(choice)

        return record


    def _positive_neighbors(self):
        '''Generate the random walk with the nodes indices
        instead of nodes labels.

        Returns
        -------
        positive_neighbors : 2d-array
            array.shape = (steps, number of nodes)
        '''
        random_walks = self.random_walk()
        positive_neighbors = np.array([
            np.nonzero(random_walks[i][:,None] == self.graph.nodes)[1]
            for i in range(self.steps)])
        return positive_neighbors


    def _negative_table(self):
        '''Table with nodes which appear with a frequency
        proportional to their degree to the 3/4.

        Returns
        -------
        table : 1d-array
        '''
        negative_table_parameter = 5
        degrees = self.edge_pivot.out.apply(
            lambda x: negative_table_parameter * int(len(x)**(3./4.))).values
        nodes = self.edge_pivot.index.values

        return np.repeat(nodes,degrees)


    def _negative_neighbors(self):
        '''Generate negative neighbors choosing randomly
        from the negative table. Node with higher degree 
        will appear more frequently.

        Returns
        -------
        negative_neighbors : 2d-array
            array.shape = (steps, number of nodes)
        '''
        negative_table = self._negative_table()
        negative_index = np.random.randint(
            len(negative_table),size=(self.steps,self.size))
        negative_index = negative_table[negative_index]

        negative_neighbors = np.array([
            np.nonzero(negative_index[i][:,None] == self.graph.nodes)[1]
            for i in range(self.steps)])

        return negative_neighbors


    def _grad_squared_distance(self,point_a,point_b):
        '''Gradient of squared hyperbolic distance.

        Gradient of the squared distance based on the
        Ball representation according to point_a

        Parameters
        ----------
        point_a : array-like, shape = [n_samples,dim]
            First point ini hyperbolic space.
        point_b : array-like, shape = [n_samples,dim]
            Second point in hyperbolic space.

        Returns
        -------
        dist : array-like, shape = [n_samples,1]
            Geodesic squared distance between the two points.
        '''
        hyperbolic_metric = PoincareBall(2).metric
        log_map = hyperbolic_metric.log(point_b,point_a)

        return -2 * log_map


    def _log_sigmoid(self,vector):
        '''Logsigmoid function.
    
        Apply log sigmoid function

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dim]

        Returns
        -------
        result : array-like, shape=[n_samples, dim]
        '''
        return np.log((1 / (1 + np.exp(-vector))))


    def _grad_log_sigmoid(self,vector):
        '''Gradient of log sigmoid function.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dim]

        Returns
        -------
        gradient : array-like, shape=[n_samples,dim]
        '''
        return 1 / (1 + np.exp(vector))


    def loss(self,nodes_embedding_temp,pos_neighbors_temp,neg_neighbors_temp):
        '''Compute the mean loss of all the nodes and
        the grad by which each node needs to move.

        Parameters
        ----------
        nodes_embedding_temp : 1d-array
            indices of current nodes
        pos_neighbors_temp : 1d-array
            indices of positive neighbors
        neg_neighbors_temp : 2d-array
            indices of two sets of negative neighbors

        Returns
        -------
        loss : float
            the mean loss
        grad : 2d-array
            the gradient for all the nodes and coordinates
            array.shape = (number of nodes, coordinates)
        '''
        nodes_embedding = self.embeddings[nodes_embedding_temp]
        positive_neighbors = self.embeddings[pos_neighbors_temp]
        negative_neighbors = self.embeddings[neg_neighbors_temp]

        positive_distance = self.manifold.metric.squared_dist(
            nodes_embedding,positive_neighbors)
        positive_loss = self._log_sigmoid(-positive_distance)

        negative_distance = self.manifold.metric.squared_dist(
            nodes_embedding,negative_neighbors)
        negative_loss = self._log_sigmoid(negative_distance)
        total_loss = - np.mean(positive_loss + np.sum(negative_loss,axis=0))

        positive_log_sigmoid_grad = - self._grad_log_sigmoid(
            -positive_distance)
        positive_distance_grad = self._grad_squared_distance(
            nodes_embedding,positive_neighbors)
        positive_grad = np.stack(
            (positive_log_sigmoid_grad,positive_log_sigmoid_grad),axis=-1)* \
            positive_distance_grad

        negative_distance_grad = self._grad_squared_distance(
            nodes_embedding, negative_neighbors)
        negative_distance_grad = np.transpose(negative_distance_grad,(1,0,2))

        negative_log_sigmoid_grad = self._grad_log_sigmoid(negative_distance)
        negative_log_sigmoid_grad = np.transpose(
            negative_log_sigmoid_grad)[...,None]

        negative_grad = negative_distance_grad * negative_log_sigmoid_grad
        negative_grad = np.transpose(negative_grad,(1,0,2))
        grad = - (positive_grad + np.sum(negative_grad,axis=0))

        return total_loss,grad


    def get_embedding(self,return_loss=None,images=None):
        '''Get the new embedded nodes.
        This runs over many epochs to get the best embedding of the nodes
        into the hyperbolic space.

        Returns
        -------
        embedding : 2d-array
            array.shape = (number of nodes, coordinates)
        '''
        positive_neighbors = self._positive_neighbors()
        negative_neighbors = self._negative_neighbors()
        c=0
        for epoch in range(self.max_epochs):
            for step in range(self.steps-self.n_negative):
                #get indices of current nodes, positive & negative neighbors
                nodes_embedding_temp = positive_neighbors[step]
                pos_neighbors_temp = positive_neighbors[step+1]
                neg_neighbors_temp = np.array(
                    negative_neighbors[step:step+self.n_negative])

                #compute loss and grad
                loss,grad = self.loss(
                    nodes_embedding_temp,pos_neighbors_temp,neg_neighbors_temp)

                #update grad on repeated indices and update nodes' coordinates
                grad_temp = np.zeros(grad.shape)
                np.add.at(grad_temp,nodes_embedding_temp,grad)
                indices_to_update,indices_counts = np.unique(
                    nodes_embedding_temp,return_counts=True)

                grad_temp[indices_to_update] /= np.stack(
                    (indices_counts,indices_counts),axis=-1)

                self.embeddings[indices_to_update] = self.manifold.metric.exp(
                        -self.lr * grad_temp[indices_to_update],
                        self.embeddings[indices_to_update])
                #logging.info(f'epoch:{epoch},step:{step},loss:{loss}')
               
                if images:
                    if step%2 ==0:
                        print(f'img:{c},saving images...')
                        self.save_img(c)
                        c+=1

        if return_loss:
            return loss
        else:
            return 

    def save_img(self,step):
        fig,ax = plt.subplots(figsize=(20,10))
        degrees = self.edge_pivot.out.apply(lambda x: len(x)*10).values
        
        ax.scatter(self.embeddings[~self.graph.final.data,0],
            self.embeddings[~self.graph.final.data,1],
            10*np.log(degrees[~self.graph.final.data]),
            alpha=1,color='#1AF8FF')
        ax.scatter(self.embeddings[self.graph.final.data,0],
            self.embeddings[self.graph.final.data,1],
            10*np.log(degrees[self.graph.final.data]),
            alpha=1,color='#FF1AE3')
        
        '''
        ax.scatter(self.embeddings[:,0],self.embeddings[:,1],
            alpha=1,color='#1AF8FF',s=50)
        for edge in self.graph.edges:
            x0,y0 = self.embeddings[edge[0]]
            x1,y1 = self.embeddings[edge[1]]
            ax.plot([x0,x1],[y0,y1],alpha=0.3,color='#FF1AE3')
        '''
        disk=Circle((0,0),1,color='white',fill=False)
        ax.add_patch(disk)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        plt.savefig('images/frame_'+str(int(step))+'.png')
        plt.close()
