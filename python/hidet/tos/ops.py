from typing import Sequence, Union, Callable, Any
from hidet.tos.module import Operator, Tensor
from hidet.ir.layout.data_layout import DataLayout, StridesLayout, RowMajorLayout, ColumnMajorLayout
from hidet.ir.primitives import cuda_max, cuda_sqrt, cuda_rsqrt
from hidet.ir.dialects.compute import compute
from hidet import tasks
from hidet.ir.task import Task
from hidet.utils import prod


class Conv2dOp(Operator):
    def __init__(self,
                 input: Tensor,
                 weight: Tensor,
                 padding: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]]
                 ):
        assert input.shape[1] == weight.shape[1], 'in_channels does not match'
        inputs = [input, weight]
        batch_size, in_channels, height, width = input.shape
        out_channels = weight.shape[0]
        kernel = weight.shape[2:]
        task = tasks.nn.conv2d(batch_size, in_channels, height, width, out_channels, kernel, padding, stride)
        super().__init__(inputs, task)


class MatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        assert len(a.shape) == len(b.shape) == 2, a.shape[1] == b.shape[0]
        task = tasks.nn.matmul(a.shape[0], b.shape[1], a.shape[1])
        super().__init__([a, b], task)


class MaxPool2dOp(Operator):
    def __init__(self,
                 input: Tensor,
                 kernel: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]]
                 ):
        inputs = [input]
        task = tasks.nn.max_pool2d(shape=input.shape, kernel=kernel, strides=stride, padding=padding)
        super().__init__(inputs, task)


class AvgPool2dOp(Operator):
    def __init__(self,
                 input: Tensor,
                 kernel: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]]
                 ):
        inputs = [input]
        task = tasks.nn.avg_pool2d(shape=input.shape, kernel=kernel, strides=stride, padding=padding)
        super().__init__(inputs, task)


class UnaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, op, name: str):
        super().__init__(inputs=[x], task=tasks.nn.unary_elementwise(name, x.shape, op=op))


class BinaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, y: Tensor, op, name: str, z_layout=None):
        super().__init__(inputs=[x, y], task=tasks.nn.binary_elementwise(name, x.layout, y.layout, op=op, z_layout=z_layout))


class ReluOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: cuda_max(v, 0), name='relu')


class SqrtOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: cuda_sqrt(v), name='sqrt')


class RsqrtOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: cuda_rsqrt(v), name='rsqrt')


class AddOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a + b, name='add')


class SubOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a - b, name='add')


class MultiplyOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a * b, name='add')


class DivideOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a / b, name='add')


class ReshapeOp(Operator):
    def __init__(self, x: Tensor, shape):
        size = prod(x.shape)
        shape = self.normalize_shape(shape, size)
        if isinstance(x.layout, (RowMajorLayout, ColumnMajorLayout)):
            layout = RowMajorLayout(shape) if isinstance(x.layout, RowMajorLayout) else ColumnMajorLayout(shape)
        else:
            layout = RowMajorLayout(shape=shape)
        task = tasks.copy(src_layout=x.layout, dst_layout=layout)
        super().__init__([x], task)

    @staticmethod
    def normalize_shape(shape, size):
        cnt = sum([1 for v in shape if v == -1])
        if cnt == 0:
            assert prod(shape) == size
            return shape
        elif cnt == 1:
            remain_size = prod([v for v in shape if v != -1])
            assert size % remain_size == 0
            return [v if v != -1 else size // remain_size for v in shape]
        else:
            raise ValueError()


def conv2d(input: Tensor, weight, padding, stride) -> Tensor:
    return Conv2dOp(input, weight, padding, stride).outputs[0]


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatmulOp(a, b).outputs[0]


def max_pool2d(input: Tensor, kernel, stride, padding) -> Tensor:
    return MaxPool2dOp(input, kernel, stride, padding).outputs[0]


def avg_pool2d(input: Tensor, kernel, stride, padding) -> Tensor:
    return AvgPool2dOp(input, kernel, stride, padding).outputs[0]


def relu(input) -> Tensor:
    return ReluOp(input).outputs[0]


def add(x: Tensor, y: Tensor) -> Tensor:
    return AddOp(x, y).outputs[0]


def sub(x: Tensor, y: Tensor) -> Tensor:
    return SubOp(x, y).outputs[0]


def multiply(x: Tensor, y: Tensor) -> Tensor:
    return MultiplyOp(x, y).outputs[0]


def divide(x: Tensor, y: Tensor) -> Tensor:
    return DivideOp(x, y).outputs[0]


def sqrt(x: Tensor) -> Tensor:
    return SqrtOp(x).outputs[0]


def rsqrt(x: Tensor) -> Tensor:
    return RsqrtOp(x).outputs[0]


def reshape(x: Tensor, shape) -> Tensor:
    return ReshapeOp(x, shape).outputs[0]


def flatten(x: Tensor, start_dim=0, end_dim=-1) -> Tensor:
    rank = len(x.shape)
    shape = []
    start_dim = (rank + start_dim) % rank
    end_dim = (rank + end_dim) % rank
    for i, dim in enumerate(x.shape):
        if i <= start_dim or i > end_dim:
            shape.append(dim)
        else:
            shape[-1] = shape[-1] * dim
    return reshape(x, shape)


def batch_norm_infer(x: Tensor, running_mean: Tensor, running_var: Tensor, epsilon=1e-5, axis=1):
    assert len(x.shape) == 4 and axis == 1
    assert len(running_mean.shape) == 1 and len(running_var.shape) == 1
    assert x.shape[1] == running_mean.shape[0] == running_var.shape[0]
    n, c, h, w = x.shape
    running_mean = running_mean.reshape([1, c, 1, 1])
    running_var = running_var.reshape([1, c, 1, 1])
    return (x - running_mean) * (running_var + epsilon).rsqrt()

