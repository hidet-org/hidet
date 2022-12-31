# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, List
import math
from hidet.graph import ops
from hidet.graph.common import normalize
from hidet.graph.module import Module, Tensor
from hidet.graph.tensor import randn, zeros, ones
from hidet.graph.modules.container import Sequential, ModuleList  # pylint: disable=unused-import


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = normalize(kernel_size)
        self.padding = normalize(padding)
        self.stride = normalize(stride)
        self.groups = groups
        self.weight = randn(
            shape=[out_channels, in_channels, *self.kernel], dtype='float32', stddev=1.0 / math.sqrt(out_channels)
        )

    def extra_str(self) -> str:
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel, self.stride, self.padding
        )

    def forward(self, x):
        x = ops.pad(x, ops.utils.normalize_padding(self.padding))
        return ops.conv2d(x, self.weight, self.stride, self.groups)


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


class Relu(Module):
    def forward(self, x):
        return ops.relu(x)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding

    def extra_str(self) -> str:
        return 'kernel_size={}, stride={}, padding={}'.format(self.kernel, self.stride, self.padding)

    def forward(self, x):
        return ops.max_pool2d(x, self.kernel, self.stride, self.padding)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding

    def extra_str(self) -> str:
        return 'kernel_size={}, stride={}, padding={}'.format(self.kernel, self.stride, self.padding)

    def forward(self, x):
        return ops.avg_pool2d(x, self.kernel, self.stride, self.padding)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = normalize(output_size)
        assert tuple(self.output_size) == (1, 1), 'current only support this'

    def extra_str(self) -> str:
        return 'output_size={}'.format(self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        return ops.avg_pool2d(x, kernel=(h, w), stride=(1, 1), padding=(0, 0))


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = randn(shape=[num_embeddings, embedding_dim], dtype='float32', mean=0.0, stddev=1.0)

    def forward(self, indices: Tensor) -> Tensor:
        return ops.take(self.weight, indices, axis=0)


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
        if self.weight:
            x = x * self.weight
        if self.bias:
            x = x + self.bias
        return x


class Gelu(Module):
    def forward(self, x):
        return x * (ops.erf(x * (1.0 / 1.4142135381698608)) + 1.0) * 0.5


class Tanh(Module):
    def forward(self, x):
        return ops.tanh(x)
