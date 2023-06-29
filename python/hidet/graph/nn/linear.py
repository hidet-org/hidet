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
from typing import Optional
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

    def extra_str(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        # x = ops.matmul(x, ops.transpose(self.weight)) # will duplicate weight memory consumption
        # workaround: use ops.matmul with some transformations
        # todo: use matmul(..., trans_a=False, trans_b=True) when we have support for transposed matmul

        out_shape = list(x.shape[:-1]) + [self.out_features]
        x = ops.reshape(x, [-1, self.in_features]).transpose(0, 1)
        x = ops.matmul(self.weight, x).transpose(0, 1).reshape(out_shape)

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


class SymQuantLinearTransposed(Module):
    def __init__(self, weight: Tensor, bias: Optional[Tensor] = None, quant_type: str = 'int8'):
        super().__init__()
        self.in_features = weight.shape[0]
        self.out_features = weight.shape[1]
        qweight, scale = ops.symmetric_quantize(weight, quant_type=quant_type, dims=[-1])
        self.qweight = qweight
        self.scale = scale
        self.bias = bias

    def extra_str(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = ops.matmul(x, ops.symmetric_dequantize(ops.barrier(self.qweight), self.scale, dims=[-1]))
        if self.bias is not None:
            x = ops.add(x, self.bias)
        return x
