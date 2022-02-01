import numpy as np
import random
import networkx as nx

import logging
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Circle, ConnectionPatch
plt.style.use('dark_background')

#from jet_tools import Components
import geomstats
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_ball import PoincareBall


class HyperbolicEmbedding:
  def __init__(self,dim=2,max_epochs=50,lr=.05,n_negative=2,context_size=1):
    self.dim = dim
    self.max_epochs = max_epochs
    self.lr = lr
    self.n_negative = n_negative
    self.context_size = context_size	

  def _walk(self,G,node,walk_length):
    '''
    Generate a random walk.

    Parameters
    ----------
    G : networkx.Graph
      A networkx graph
    node : int
      Starting point
    walk_length : int
      Number of steps

    Returns
    -------
    walk : 1d array
      random walk;
      First index is first step     
    '''

    path_temp = [node]
    for i in range(walk_length):
      current_step = path_temp[-1]
      next_step = random.choice(list(G.neighbors(current_step)))
      path_temp.append(next_step)

    return np.array(path_temp)		


  def random_walk(self,G,walk_length):
    '''
    Compute a set of random walk on a Graph.

    For each node of the graph, generates a number of
    radom walks of a specified length.
    Two consecutive nodes in the random walk are necessarily
    related with an edge. The walks capture the structure of the graph.

    Parameters
    ----------
    G : networkx.Graph
      A networkx graph.
    walk_length : int
      Length of a random walk in terms of number of edges.

    Returns
    -------
    random_walk : 2d array
      An array of a random walk for each node.
    '''
    path = []
    for node in G.nodes:
      path.append(self._walk(G,node,walk_length))

    return np.array(path)


  def grad_squared_distance(self,point_a,point_b):
    '''
    Gradient of squared hyperbolic distance.
    
    Gradient of the squared distance based on the 
    Ball representation according to point_a

    Parameters
    ----------
    point_a : array-like, shape = [n_samples,dim]
      First point in hyperbolic space.
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


  def log_sigmoid(self,vector):
    '''
    Logsigmoid function.

    Apply log sigmoid function

    Parameters
    ----------
    vector : array-like, shape=[n_samples, dim]

    Returns
    -------
    result : array-like, shape=[n_samples, dim]
    '''
    return np.log((1 / (1 + np.exp(-vector))))


  def grad_log_sigmoid(self,vector):
    '''
    Gradient of log sigmoid function.

    Parameters
    ----------
    vector : array-like, shape=[n_samples, dim]

    Returns
    -------
    gradient : array-like, shape=[n_samples, dim]
    '''
    return 1 / (1 + np.exp(vector))


  def loss(self,example_embedding, context_embedding, negative_embedding,manifold):
    '''
    Compute loss and grad.

    Compute loss and grad given embedding of the current example,
    embedding of the context and negative sampling embedding.
    '''
    n_edges, dim = negative_embedding.shape[0], example_embedding.shape[-1]
    example_embedding = np.expand_dims(example_embedding, 0)
    context_embedding = np.expand_dims(context_embedding, 0)

    positive_distance = manifold.metric.squared_dist(example_embedding, context_embedding)
    positive_loss = self.log_sigmoid(-positive_distance)

    reshaped_example_embedding = np.repeat(example_embedding, n_edges, axis=0)
    negative_distance = manifold.metric.squared_dist(reshaped_example_embedding, negative_embedding)
    negative_loss = self.log_sigmoid(negative_distance)

    total_loss = -(positive_loss + negative_loss.sum())

    positive_log_sigmoid_grad = - self.grad_log_sigmoid(-positive_distance)
    positive_distance_grad = self.grad_squared_distance(example_embedding, context_embedding)
    positive_grad = np.repeat(positive_log_sigmoid_grad, dim, axis=-1) * positive_distance_grad

    negative_distance_grad = self.grad_squared_distance(reshaped_example_embedding, negative_embedding)
    negative_distance = gs.to_ndarray(negative_distance,to_ndim=2, axis=-1)
    negative_log_sigmoid_grad = self.grad_log_sigmoid(negative_distance)
    negative_grad = negative_log_sigmoid_grad * negative_distance_grad

    example_grad = - (positive_grad + negative_grad.sum(axis=0))

    return total_loss, example_grad


  def save_img(self,df,G,data,epoch,loss=None):
    '''
    Save image of the Poincare Disk and the Phase Space.
   
    Parameters
    ----------
    df : pandas.DataFrame
      Particles' dataframe
    G : networkx.Graph
      A Networkx graph
    data : 2d array
      Coordinates of the nodes in the hyperbolic space
    epoch : int 
      epoch index
    loss : float
      averaged loss 

    Returns
    -------
    '''
    plt.style.use('dark_background')
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    ax = ax.flatten()
    fig.patch.set_facecolor('#151515')
    fig.patch.set_alpha(1) 

    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)

    idxs = idxs = [51,52,71,72] #which I want to highlight
    hyperbolic = Hyperboloid(dim=2, coords_type='extrinsic')
    for i_embedding, embedding in enumerate(data):
        x = embedding[0]
        y = embedding[1]
        node = list(G.nodes)[i_embedding]
        if G.degree[i_embedding] < 2:
            if i_embedding in idxs:
                ax[0].scatter(x,y,alpha=1,color='#DF0101',s = 50)
            else:
                ax[0].scatter(x,y,alpha=1,color='#1AF8FF',s = 50)
        else:
            ax[0].scatter(x,y,alpha=1,color='#FF1AE3',s = 50)

        pt_id = i_embedding
        #ax[0].annotate(pt_id, (x+0.01,y))

    for edge in G.edges:
        start,end = edge
        x1,y1 = data[start]
        x2,y2 = data[end]
        #plt.plot([x1,x2],[y1,y2],c='r')    

        initial_point = hyperbolic.from_coordinates(np.array([x1,y1]), 'intrinsic')
        end_point = hyperbolic.from_coordinates(np.array([x2,y2]), 'intrinsic')

        geodesic = hyperbolic.metric.geodesic(
            initial_point=initial_point, end_point=end_point)

        points = geodesic(np.linspace(0., 1., 10))    
        ax[0].plot(points[:,1],points[:,2],alpha=0.3,c='#FFF51A')


    disk = Circle((0, 0), 1, color='white', fill=False)
    ax[0].add_patch(disk)
    ax[0].text(0.7,0.8,f'loss: {loss}')
    ax[0].text(0.7,0.85,f'iteration: {epoch}')

    #ax.legend()
    ax[0].add_patch(disk)
    ax[0].set_xlim(-1,1)
    ax[0].set_ylim(-1,1)
    ax[0].set_title('Poincare disk',fontsize=40)


    finals = df.index[df['final']==True]
    ax[1].scatter(df['eta'][finals],df['phi'][finals],s=50,c='#1AF8FF')
    #for p in finals:
    #    ax[1].annotate(df['p_index'][p],(df['eta'][p]+0.01,df['phi'][p]))
    ax[1].set_xlim(-5,5)
    ax[1].set_ylim(-np.pi,np.pi)
    ax[1].set_xlabel('eta',fontsize=22)
    ax[1].set_ylabel('phi',fontsize=22)
    ax[1].set_title('Phase space',fontsize=40)

    '''
    for idx00 in idxs:
        xy_left = data[idx00]
        xy_right = (df['eta'][df.index[df['p_index'] == idx00][0]],df['phi'][df.index[df['p_index'] == idx00][0]])

        ax[0].scatter(xy_left[0],xy_left[1],alpha=1,color='#DF0101',s = 50)
        ax[1].scatter(xy_right[0],xy_right[1],s=50,c='#DF0101') #change color
        con = ConnectionPatch(xyA=xy_left, coordsA=ax[0].transData,
                              xyB=xy_right, coordsB=ax[1].transData,color='#DF0101')
        fig.add_artist(con)
    '''
    plt.savefig('images/frame_'+str(int(epoch/2))+'.png')
    plt.close()


  def make_video(self,frames=50):
    '''
    Make video from images
    '''
    with imageio.get_writer('images/OneEpisode.mp4', fps=10) as writer:
        for i in range(0,frames):
            image = imageio.imread('images/frame_'+str(int(i))+'.png')
            writer.append_data(image)
        

  def get_embedding(self,G,df,create_image=None,print_info=True):
    hyperbolic_manifold = PoincareBall(2)
    
    embeddings = np.random.normal(size=(G.number_of_nodes(),self.dim))
    embeddings = embeddings * 0.02

    #generating the random walk
    random_walks = self.random_walk(G=G,walk_length=5) 

    #creating negative table
    negative_table_parameter = 7
    negative_sampling_table = []
    for idx,degree in list(G.degree()):
      negative_sampling_table += ([idx] * int((degree**(3./4))) * negative_table_parameter)
    negative_sampling_table = np.array(negative_sampling_table)

    #starting the process
    for epoch in range(self.max_epochs):
      total_loss = []
      for path in random_walks:
        for example_index, one_path in enumerate(path):
          context_index = path[max(0, example_index - self.context_size):min(example_index + self.context_size,len(path))]
          negative_index = np.random.randint(negative_sampling_table.shape[0],size=(len(context_index),self.n_negative))
          negative_index = negative_sampling_table[negative_index]

          example_embedding = embeddings[one_path]
          for one_context_i, one_negative_i in zip(context_index,negative_index):
            context_embedding = embeddings[one_context_i]
            negative_embedding = embeddings[one_negative_i]
            
            l, g_ex = self.loss(example_embedding,context_embedding,negative_embedding,hyperbolic_manifold)
            total_loss.append(l)

            example_to_update = embeddings[one_path]
            embeddings[one_path] = hyperbolic_manifold.metric.exp(-self.lr * g_ex, example_to_update)
      
      if create_image:      
        if epoch%2 == 0:
            loss_temp = float(sum(total_loss, 0) / len(total_loss))
            self.save_img(df,G,embeddings,epoch=epoch,loss = round(loss_temp,3))
      
      if print_info == True:      
        logging.info('iteration %d loss_value %f',epoch, sum(total_loss, 0) / len(total_loss))    

    #updating the graph
    for i in range(G.number_of_nodes()):
      G.nodes[i]['hyper_coords'] = embeddings[i] #saving nodes attributes
      
      for j in G.neighbors(i):
        hyper_distance = np.sqrt(hyperbolic_manifold.metric.squared_dist(embeddings[i],embeddings[j]))
        G[i][j]['hyper_dist'] = hyper_distance #saving edges attributes

    return G
