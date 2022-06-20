from abc import ABC, abstractmethod
from torch.nn import Parameter
from torch import Tensor

class Manifold(ABC):
    @abstractmethod
    def sqdist(self, u: Tensor, v: Tensor) -> Tensor:
        '''
        Distance function
        '''
        raise NotImplementedError

    @abstractmethod
    def expmap(self, p: Tensor, d_p: Tensor) -> Tensor:
        '''
        Exponential map
        '''
        raise NotImplementedError

    @abstractmethod
    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        '''
        Logarithmic map
        '''
        raise NotImplementedError

    @abstractmethod
    def inner(self, x: Tensor, u: Tensor, v=None) -> Tensor:
        '''
        Inner product
        '''
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        '''
        Parallel transport
        '''
        raise NotImplementedError


class ManifoldParameter(Parameter):
    def __new__(cls, data, requires_grad, manifold):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold):
        self.manifold = manifold

