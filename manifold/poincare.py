import torch as th
import torch.linalg as alg
from manifold import Manifold

class PoincareManifold(Manifold):
    def __init__(self):
        self.epsilon = 1e-12
        

    def _clip_vectors(self,vectors):
        '''Clip vectors to have a norm of less than one.

        Parameters
        ----------
        vectors : Tensor-like
            Can be a 1-D or 2-D (in this case the norm of each row is checked)

        Returns
        -------
        Tensor-like
            Tensor with norms clipped below 1.
        '''
        thresh = 1.0 - self.epsilon
        one_d = len(vectors.shape) == 1
        if one_d:
            norm = alg.norm(vectors)
            if norm < thresh:
                return vectors
            else:
                return thresh * vectors / norm
        else:
            norms = alg.norm(vectors,axis=1)
            if (norms < thresh).all():
                return vectors
            else:
                vectors[norms >= thresh] *= \
                    (thresh / norms[norms >= thresh])[:,None]
                return vectors
                

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
        point_a = self._clip_vectors(point_a)
        point_b = self._clip_vectors(point_b)

        sq_norm_a = alg.norm(point_a) ** 2.
        sq_norm_b = alg.norm(point_b) ** 2.
        sq_norm_ab = alg.norm(point_a - point_b) ** 2.

        cosh_angle = 1 + 2 * sq_norm_ab / (1 - sq_norm_a) / (1 - sq_norm_b) 
        cosh_angle = th.clamp(cosh_angle, min=1.)
        dist = th.acosh(cosh_angle)
        return dist


    def exp_map(self,vector,base_point=None):
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
        if base_point is None:
            base_point = th.zeros(vector.size())

        vector = self._clip_vectors(vector)
        base_point = self._clip_vectors(base_point)

        factor = self.conformal_factor(base_point)
        norm_vector = alg.norm(vector)
        tn = th.tanh(factor * norm_vector / 2.) * vector / norm_vector
        exp = self.mobius_addition(base_point,tn)
        return exp


    def log_map(self,vector,base_point=None):
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
        if base_point is None:
            base_point = th.zeros(vector.size())

        vector = self._clip_vectors(vector)
        base_point = self._clip_vectors(base_point)

        factor = self.conformal_factor(base_point)
        mobius_sum = self.mobius_addition(-base_point,vector)
        norm_sum = alg.norm(mobius_sum)
        norm_sum = self._clip_vectors(norm_sum)
        log = th.arctanh(norm_sum) * mobius_sum / norm_sum
        log *= (2. / factor)

        return log


    def conformal_factor(self,x):
        '''Return the conformal factor
        '''
        x = self._clip_vectors(x)
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
        v1 = self._clip_vectors(v1)
        v2 = self._clip_vectors(v2)

        v1v2 = th.dot(v1,v2)
        norm_v1 = alg.norm(v1) ** 2.
        norm_v2 = alg.norm(v2) ** 2.
        
        numerator = (1. + 2.*v1v2 + norm_v2) * v1 + (1. - norm_v1)*v2
        denominator = 1. + 2.*v1v2 + norm_v1*norm_v2
        addition = numerator/denominator
        return addition


    def scalar_product(self,vector,scalar):
        '''Product of a vector with a scalar.

        Parameters
        ----------
        vector : Tensor-like, shape=[m,1]
            vector
        scalar : float
            scalar value

        Returns
        -------
        scaled_vector : Tensor-like, shape=[m,1]
            vector multiplied by the scalar
        '''
        vector = self._clip_vectors(vector)

        log = self.log_map(vector)
        scaled_vector = self.exp_map(r * log)
        return scaled_vector


    def parallel_transport(self,vector,end_point,start_point=None):
        '''Parallel transport of a vector from a starting point to
        an end point.

        Parameters
        ----------
        vector : Tensor-like, shape[...,2]
            vector to be transported
        end_point : Tensor-like, shape[...,2]
            end point
        start_point : Tensor-like, shape[...,2]
            start_point - by default the center of the disk

        Returns
        -------
        transported_vec : Tensor-like, shape[...,2]
            the new vector after transported
        '''
        if start_point is None:
            start_point = th.zeros(vector.size())

        vector = self._clip_vectors(vector)
        start_point = self._clip_vectors(start_point)
        end_point = self._clip_vectors(end_point)

        exp = self.exp_map(vector,start_point)
        mobius_sum = self.mobius_addition(end_point,exp)
        transported_vec = self.log_map(mobius_sum,end_point)
        return transported_vec


    def matrix_vector_multiplication(self,matrix,vector):
        '''Multiplication between a matrix and a vector in the  
        Poincare ball.

        Parameters
        ----------
        matrix : Tensor-like, shape=[m,n]
            matrix, typically the weight matrix
        vector : Tensor-like, shape=[n,1]
            vector

        Returns
        -------
        movius_mult : Tensor-like, shape[m,1]
            new vector
        '''
        matrix = self._clip_vectors(matrix)
        vector = self._clip_vectors(vector)

        norm_vector = alg.norm(vector)
        mult = th.matmul(matrix,vector)
        norm_mult = alg.norm(mult)
        mobius_mult = th.tanh(norm_mul / norm_vector * \
                th.arctanh(norm_vector) ) * mult / norm_mult
        return mobius_mult


    def bias_translation(self,vector,bias):
        '''Translation of a vector by a bias.

        Parameters
        ----------
        vector : Tensor-like, shape=[m,1]
            vector to be translated
        bias : Tensor-like, shape=[m,1]
            vector to apply

        Returns
        -------
        translated_vec : Tensor-like, shape=[m,1]
            translated vector
        '''
        vector = self._clip_vectors(vector)
        bias = self._clip_vectors(bias)

        factor_vector = self.conformal_factor(vector)
        factor_bias = self.conformal_factor(bias)
        factor = factor_vector / factor_bias
        log_vector = self.log_map(bias)
        translated_vec = self.exp_map(factor * log_vector, vector)
        return translated_vec
