# hyperTree project.
This project aims to embed the tree structure of a particles' shower into the 
hyperbolic space, in particular the Poincare' disk. 
This might enhance the clustering algorithms as the tree structure is well 
preserved in the new space.

# Features:
- hyperEmbedding, takes a full event and projects it into the Poincare' disk
- geNN, graph embedding neural network. With a supervised learning method, 
it projects leaves particles into the Poincare' disk, using what has been 
learnt from the hyperEmbedding

# Milestone
The embedding in the hyperbolic space is obtained through the Word2vec 
algorithm. Once the event has been generated and the process has been showered,
the Graphicle object is passed to the algorithm. Within few epochs and with 
a random walk of ten step, a minimum can be found within 1s of run and a total
loss less than 1. 

