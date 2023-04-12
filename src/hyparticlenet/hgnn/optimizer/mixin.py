from hyparticlenet.hgnn.nn.manifold import PoincareBallManifold
import torch

class OptimMixin(object):
    _default_manifold = PoincareBallManifold()

    def __init__(self, *args, stabilize=None, **kwargs):
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def add_param_group(self, param_group: dict):
        param_group.setdefault('stabilize', self._stabilize)
        return super().add_param_group(param_group)

    def stabilize_group(self, group):
        pass

    def stabilize(self):
        '''Stabilize parameters if they are off-manifold due to numerical
        reasons.'''
        for group in self.param_groups:
            self.stabilize_group(group)
