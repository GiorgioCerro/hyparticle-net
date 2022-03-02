from poincare import PoincareManifold
import torch as th
poincare = PoincareManifold()
th.set_default_dtype(th.float64)

def test_algebraic_check():
    alpha1, alpha2 = 2. * th.pi * th.rand(2)
    r1, r2 = th.rand(1), th.rand(1)

    a = th.Tensor([th.cos(alpha1),th.sin(alpha1)]) * r1
    b = th.Tensor([th.cos(alpha2),th.sin(alpha2)]) * r2

    exp = poincare.exp_map(a,b) #exp of vec a around point b
    log = poincare.log_map(exp,b) #log of vec a around point b
    
    assert sum(th.isclose(log,a)) == len(a), \
            'Vec should be equal to log(exp(vec))'


if __name__ == '__main__':
    test_algebraic_check()
    print('Everything passed')
