"""
Weight Reparametrization to enable Networks with strictly positive weights ( but arbitrary bias )
"""
from torch.nn.parameter import Parameter
from torch.nn import Softplus, ReLU
import torch.functional as F


def _exp(input):
    return input.exp()

def _log(input):
    return input.log()


# Code for weight reparametrization is based on torch.nn.utils.weight_norm
class WeightReparametrization(object):

    def __init__(self, name, transform=_exp):
        self.name = name
        self.transform = transform

    def compute_weight(self, module):
        pre = getattr(module, self.name + '_pre')
        return self.transform(pre)

    @staticmethod
    def apply(module, name, transform=_exp):
        fn = WeightReparametrization(name, transform=transform)

        weight = module._parameters[name]

        # remove w from parameter list
        del module._parameters[name]

        # we reuse the weight Parameter for our purpose
        pre_weight = Parameter(transform(weight).data)
        module.register_parameter(name + '_pre', pre_weight)

        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_pre']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def weight_reparametrization(module, name='weight', transform=_exp):
    r"""Applies componentwise weight reparametrization to a parameter in the given module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        transform (Callable, optional): Function, transforming an input tensor componentwise, returning a tensor of same shape as input

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_reparametrization(nn.Linear(20, 40), name='weight')

    """
    WeightReparametrization.apply(module, name, transform=transform)
    return module


def remove_weight_reparametrization(module, name='weight'):
    r"""Removes the componentwise weight reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightReparametrization) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))
