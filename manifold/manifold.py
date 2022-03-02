from abc import ABC, abstractmethod

class Manifold(ABC):
    @abstractmethod
    def distance(self, u, v):
        '''
        Distance function
        '''
        raise NotImplementedError

    @abstractmethod
    def exp_map(self, p, d_p):
        '''
        Exponential map
        '''
        raise NotImplementedError

    @abstractmethod
    def log_map(self, x, y):
        '''
        Logarithmic map
        '''
        raise NotImplementedError

    @abstractmethod
    def parallel_transport(self, x, y, v):
        '''
        Parallel transport
        '''
        raise NotImplementedError
