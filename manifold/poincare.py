import numpy as np
#import numpy.linalg as alg
import torch as th
import torch.linalg as alg
from manifold import Manifold

class PoincareManifold(Manifold):
    #def __init__(self):
    #    pass
    
        
    def distance(self,point_a,point_b):
        '''Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : Tensor-like, shape=[...,2]
            First point on the disk
        point_b : Tensor-like, shape=[...,2]
            Second point on the disk

        Returns
        -------
        dist : Tensor-like, shape=[...]
            Geodesic distance between points
        '''
        sq_norm_a = alg.norm(point_a) ** 2.
        sq_norm_b = alg.norm(point_b) ** 2.
        sq_norm_ab = alg.norm(point_a - point_b) ** 2.

        cosh_angle = 1 + 2 * sq_norm_ab / (1 - sq_norm_a) / (1 - sq_norm_b) 
        cosh_angle = th.clamp(cosh_angle, min=1.)
        dist = th.acosh(cosh_angle)
        return dist


    def exp_map(self,vector,base_point=th.Tensor([0.,0.])):
        '''Compute the Riemann exponential of a tangent vector.

        Parameters
        ----------
        vector : Tensor-like, shape=[...]
            vector
        base_point : Tensor-like, shape=[...]
            base point, by default the center (0,0)

        Returns
        -------
        exp : Tensor-like, shape=[...]
            Point on the hypersphere equal to the Riemannian exponential
            of tangent vector at the base point.
        '''
        factor = self.conformal_factor(base_point)
        norm_vector = alg.norm(vector)
        tn = th.tanh(factor * norm_vector / 2.) * vector / norm_vector
        exp = self.mobius_addition(base_point,tn)
        return exp


    def log_map(self,vector,base_point=th.Tensor([0.,0.])):
        '''Compute the logarithmic map on the manifold.

        Parameters
        ----------
        vector : Tensor-like, shape=[...]
            vector
        base_point : Tensor-like, shape=[...]
            base point, by default the center (0,0)

        Returns
        -------
        log : Tensor-like, shape=[]
            Logarithmic of point at the base point.
        '''
        factor = self.conformal_factor(base_point)
        mobius_sum = self.mobius_addition(-base_point,vector)
        norm_sum = alg.norm(mobius_sum)
        norm_sum = th.clamp(norm_sum,-1.,1.)
        log = th.arctanh(norm_sum) * mobius_sum / norm_sum
        log *= (2. / factor)

        return log


    def conformal_factor(self,x):
        '''Return the conformal factor
        '''
        norm_x = alg.norm(x)
        return 2. / (1. - norm_x ** 2.)


    def mobius_addition(self,v1,v2):
        '''Addition between two vectors in the manifold.
        This is not commutative.
        Also the difference is taken as a sum with the negative vector.

        Parameters
        ----------
        v1 : Tensor-like, shape[...,2]
            left vector
        v2 : Tensor-like, shape[...,2]
            right vector

        Returns 
        -------
        addition : Tensor-like, shape=[...]
            mobius addition of v1 + v2
        '''
        v1v2 = th.dot(v1,v2)
        norm_v1 = alg.norm(v1) ** 2.
        norm_v2 = alg.norm(v2) ** 2.
        
        numerator = (1. + 2.*v1v2 + norm_v2) * v1 + (1. - norm_v1)*v2
        denominator = 1. + 2.*v1v2 + norm_v1*norm_v2
        addition = numerator/denominator
        return addition

    def scalar_product():
        return

    def parallel_transport():
        return

