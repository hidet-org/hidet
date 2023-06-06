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
from hidet.graph.common import normalize
from hidet.graph.nn.module import Module
from hidet.graph.tensor import Tensor


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
