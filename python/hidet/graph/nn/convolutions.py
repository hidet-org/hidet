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
from hidet.graph.tensor import empty


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = normalize(kernel_size)
        self.padding = normalize(padding)
        self.stride = normalize(stride)
        self.groups = groups
        self.weight = empty(shape=[out_channels, in_channels, *self.kernel], dtype='float32')
        # use shape (oc, 1, 1) for broadcast
        self.bias = empty(shape=[out_channels, 1, 1], dtype="float32") if bias else None

    def extra_str(self) -> str:
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel, self.stride, self.padding
        )

    def forward(self, x):
        x = ops.pad(x, ops.utils.normalize_padding(self.padding))
        x = ops.conv2d(x, self.weight, stride=self.stride, groups=self.groups)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        return x
