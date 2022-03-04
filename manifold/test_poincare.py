from poincare import PoincareManifold
import torch as th
from torch.testing import assert_close
poincare = PoincareManifold()
th.set_default_dtype(th.float64)
import numpy as np

#from geomstats.geometry.poincare_ball import PoincareBall
#hyperbolic_manifold = PoincareBall(2)

def random_point():
    alpha = 2. * th.pi * th.rand(1)
    r = th.rand(1)
    point = th.Tensor([th.cos(alpha),th.sin(alpha)]) * r
    return point


def test_algebraic_check():
    point_a = th.rand(3)
    point_b = th.zeros(3)
    exp = poincare.exp_map(point_a,point_b) #exp of vec a around point b
    log = poincare.log_map(exp,point_b) #log of vec a around point b
    #log = th.Tensor(hyperbolic_manifold.metric.log(np.array(exp),np.array(point_b)))
    #print(point_a,log)
    
    assert_close(log,point_a, msg= \
            f'Vec should be equal to log(exp(vec)), a:{point_a},b:{point_b}')


if __name__ == '__main__':
    test_algebraic_check()
    print('Everything passed')
