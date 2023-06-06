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

from hidet.graph import ops
from hidet.graph.nn.module import Module
from hidet.graph.tensor import Tensor, empty


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.running_mean = empty(shape=[num_features])
        self.running_var = empty(shape=[num_features])

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
            self.weight = empty(normalized_shape)
            self.bias = empty(normalized_shape)
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
