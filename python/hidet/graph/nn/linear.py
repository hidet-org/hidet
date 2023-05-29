import math

from hidet.graph import ops
from hidet.graph.nn.module import Module
from hidet.graph.tensor import Tensor, randn, zeros


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = randn(shape=[in_features, out_features], stddev=1.0 / math.sqrt(in_features))
        if bias:
            self.bias = zeros(shape=[out_features])
        else:
            self.bias = None

    def extra_str(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = ops.matmul(x, self.weight)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        return x
