import math

from hidet.graph import ops
from hidet.graph.common import normalize
from hidet.graph.nn.module import Module
from hidet.graph.tensor import randn
from hidet.utils import prod


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
            shape=[out_channels, in_channels, *self.kernel],
            dtype='float32',
            stddev=1.0 / math.sqrt(in_channels * prod(self.kernel)),
        )

    def extra_str(self) -> str:
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel, self.stride, self.padding
        )

    def forward(self, x):
        x = ops.pad(x, ops.utils.normalize_padding(self.padding))
        return ops.conv2d(x, self.weight, self.stride, self.groups)
