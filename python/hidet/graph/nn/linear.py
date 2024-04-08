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
from hidet.graph import ops
from hidet.graph.nn.module import Module
from hidet.graph.tensor import Tensor, empty


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = empty(shape=[out_features, in_features])
        if bias:
            self.bias = empty(shape=[out_features])
        else:
            self.bias = None

        self._transposed_weight = None

    def extra_str(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    def transposed_weight(self) -> Tensor:
        if self._transposed_weight is None:
            self._transposed_weight = ops.transpose(self.weight, [1, 0])  # [in_features, out_features]
            self.weight = None
        return self._transposed_weight

    def forward(self, x: Tensor) -> Tensor:
        x = ops.matmul(x, self.transposed_weight())
        if self.bias is not None:
            x = ops.add(x, self.bias)
        return x


class LinearTransposed(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = empty(shape=[in_features, out_features])
        if bias:
            self.bias = empty(shape=[out_features])
        else:
            self.bias = None

    def extra_str(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = ops.matmul(x, self.weight)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        return x
