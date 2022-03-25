from .graph import Module, Tensor
from . import ops
from .common import normalize
from hidet.runtime import randn


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.kernel = normalize(kernel_size)
        self.padding = normalize(padding)
        self.stride = normalize(stride)
        self.weight = Tensor(dtype='float32', shape=[out_channels, in_channels, *kernel_size], init_method='rand')

    def forward(self, x):
        return ops.conv2d(x, self.weight, self.padding, self.stride)


class BatchNorm2d(Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        n, c, h, w = x.shape


class Relu(Module):
    pass


class MaxPool2d(Module):
    pass
