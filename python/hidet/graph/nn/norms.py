from typing import Union, List

from hidet.graph import ops
from hidet.graph.nn.module import Module
from hidet.graph.tensor import Tensor, zeros, ones


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.running_mean = zeros(shape=[num_features])
        self.running_var = ones(shape=[num_features])

    def extra_str(self) -> str:
        return 'eps={}'.format(self.eps)

    def forward(self, x: Tensor):
        return ops.batch_norm_infer(x, self.running_mean, self.running_var, self.eps)


class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = ones(normalized_shape)
            self.bias = zeros(normalized_shape)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        x = ops.layer_norm(x)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x
