from __future__ import annotations
from hidet.graph.tensor import Tensor
from .interpreter import register_method


@register_method(Tensor, 'cuda')
def tensor_cuda(self: Tensor):
    return self
