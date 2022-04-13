from typing import Sequence, Union, Callable, Any, List, Optional
from collections import OrderedDict
from hidet.tos.operator import Operator
from hidet.tos.tensor import Tensor
from hidet.ir.layout.data_layout import DataLayout, StridesLayout, RowMajorLayout, ColumnMajorLayout
from hidet.ir.primitives import cuda_max, cuda_sqrt, cuda_rsqrt
from hidet import tasks
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
        super().__init__(inputs, task, padding=padding, stride=stride)


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
        super().__init__(
            inputs=[input],
            task=tasks.nn.max_pool2d(shape=input.shape, kernel=kernel, strides=stride, padding=padding),
            kernel=kernel,
            stride=stride,
            padding=padding
        )


class AvgPool2dOp(Operator):
    def __init__(self,
                 input: Tensor,
                 kernel: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]]
                 ):
        super().__init__(
            inputs=[input],
            task=tasks.nn.avg_pool2d(shape=input.shape, kernel=kernel, strides=stride, padding=padding),
            kernel=kernel,
            stride=stride,
            padding=padding
        )


class SoftmaxOp(Operator):
    def __init__(self,
                 x: Tensor,
                 axis: int = 1):
        super().__init__(
            inputs=[x],
            task=tasks.nn.softmax(x.shape, axis),
            axis=axis
        )


class ReduceMeanOp(Operator):
    def __init__(self, x: Tensor, dims: List[int], keep_dim: bool = False):
        super().__init__(
            inputs=[x],
            task=tasks.reduce_mean(x.layout, dims, keep_dim),
            dims=dims,
            keep_dim=keep_dim
        )


class UnaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, op, name: str):
        super().__init__(
            inputs=[x],
            task=tasks.nn.unary_elementwise(name, x.shape, op=op)
        )


class BinaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, y: Tensor, op, name: str):
        super().__init__(
            inputs=[x, y],
            task=tasks.nn.binary_elementwise(name, x.layout, y.layout, op=op)
        )


class ReluOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: cuda_max(v, 0.0), name='relu')


class SqrtOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: cuda_sqrt(v), name='sqrt')


class RsqrtOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: cuda_rsqrt(v), name='rsqrt')


class NegOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: -v, name='neg')


class AddOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a + b, name='add')


class SubOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a - b, name='sub')


class MultiplyOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a * b, name='mul')


class DivideOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a / b, name='div')


class ReshapeOp(Operator):
    def __init__(self, x: Tensor, shape):
        size = prod(x.shape)
        shape = self.normalize_shape(shape, size)
        if isinstance(x.layout, (RowMajorLayout, ColumnMajorLayout)):
            layout = RowMajorLayout(shape) if isinstance(x.layout, RowMajorLayout) else ColumnMajorLayout(shape)
        else:
            layout = RowMajorLayout(shape=shape)
        task = tasks.copy(src_layout=x.layout, dst_layout=layout)
        super().__init__(
            inputs=[x],
            task=task,
            shape=shape
        )

    @staticmethod
    def normalize_shape(shape, size):
        # [1, -1, 224, 224] => [1, 3, 224, 224] when size = 1 * 3 * 224 * 224
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


class SqueezeOp(Operator):
    def __init__(self, x: Tensor, dims: List[int]):
        super().__init__(
            inputs=[x],
            task=tasks.squeeze(x.layout, dims),
            dims=dims
        )

    def imperative_run(self, inputs: Optional[List[Tensor]] = None) -> List[Tensor]:
        x = inputs[0] if inputs else self.inputs[0]
        if isinstance(x.layout, (RowMajorLayout, ColumnMajorLayout)):
            shape = [int(v) for v in self.task.params[1].shape]
            layout = x.layout.__class__(shape)
            return [Tensor(shape=shape, dtype=x.dtype, device=x.device, storage=x.storage, layout=layout, trace=None)]
        else:
            return Operator.imperative_run(self)


class UnsqueezeOp(Operator):
    def __init__(self, x: Tensor, dims: List[int]):
        super().__init__(
            inputs=[x],
            task=tasks.unsqueeze(x.layout, dims),
            dims=dims
        )

    def imperative_run(self, inputs: Optional[List[Tensor]] = None) -> List[Tensor]:
        x = inputs[0] if inputs else self.inputs[0]
        if isinstance(x.layout, (RowMajorLayout, ColumnMajorLayout)):
            shape = [int(v) for v in self.task.params[1].shape]
            layout = x.layout.__class__(shape)
            return [Tensor(shape=shape, dtype=x.dtype, device=x.device, storage=x.storage, layout=layout, trace=None)]
        else:
            return Operator.imperative_run(self)


class RearrangeOp(Operator):
    def __init__(self, x: Tensor, plan: List[List[int]]):
        super().__init__(
            inputs=[x],
            task=tasks.rearrange(x.layout, plan=plan)
        )


class CastOp(Operator):
    def __init__(self, x: Tensor, dtype: str):
        super().__init__(
            inputs=[x],
            task=tasks.cast(x.layout, src_dtype=x.dtype, dst_dtype=dtype)
        )


class ConcatOp(Operator):
    def __init__(self, tensors: List[Tensor], axis: int):
        super().__init__(
            inputs=tensors,
            task=tasks.concat([tensor.layout for tensor in tensors], axis=axis),
            axis=axis
        )


class TakeOp(Operator):
    def __init__(self, data: Tensor, indices: Tensor, axis: int):
        super().__init__(
            inputs=[data, indices],
            task=tasks.take(data.layout, indices.layout, axis=axis),
            axis=axis
        )


def conv2d(input: Tensor, weight, padding, stride) -> Tensor:
    return Conv2dOp(input, weight, padding, stride).get_output(0)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatmulOp(a, b).get_output(0)


def max_pool2d(input: Tensor, kernel, stride, padding) -> Tensor:
    return MaxPool2dOp(input, kernel, stride, padding).get_output(0)


def avg_pool2d(input: Tensor, kernel, stride, padding) -> Tensor:
    return AvgPool2dOp(input, kernel, stride, padding).get_output(0)


def reduce_mean(x: Tensor, dims: List[int], keep_dim: bool = False) -> Tensor:
    return ReduceMeanOp(x, dims, keep_dim).get_output(0)


def relu(input) -> Tensor:
    return ReluOp(input).get_output(0)


def add(x: Tensor, y: Tensor) -> Tensor:
    return AddOp(x, y).get_output(0)


def sub(x: Tensor, y: Tensor) -> Tensor:
    return SubOp(x, y).get_output(0)


def multiply(x: Tensor, y: Tensor) -> Tensor:
    return MultiplyOp(x, y).get_output(0)


def divide(x: Tensor, y: Tensor) -> Tensor:
    return DivideOp(x, y).get_output(0)


def sqrt(x: Tensor) -> Tensor:
    return SqrtOp(x).get_output(0)


def rsqrt(x: Tensor) -> Tensor:
    return RsqrtOp(x).get_output(0)


def neg(x: Tensor) -> Tensor:
    return NegOp(x).get_output(0)


def reshape(x: Tensor, shape) -> Tensor:
    return ReshapeOp(x, shape).get_output(0)


def squeeze(x: Tensor, dims) -> Tensor:
    return SqueezeOp(x, dims).get_output(0)


def unsqueeze(x: Tensor, dims) -> Tensor:
    return UnsqueezeOp(x, dims).get_output(0)


def concat(tensors: List[Tensor], axis: int) -> Tensor:
    return ConcatOp(tensors, axis).get_output(0)


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


def rearrange(x: Tensor, plan: List[List[int]]) -> Tensor:
    return RearrangeOp(x, plan).get_output(0)


def softmax(x: Tensor, axis=1) -> Tensor:
    if len(x.shape) < 4:
        dims = list(range(len(x.shape), 4))
        xx = unsqueeze(x, dims)
        return SoftmaxOp(xx, axis).get_output(0).squeeze(dims)
    return SoftmaxOp(x, axis).get_output(0)


def cast(x: Tensor, dtype: str) -> Tensor:
    return CastOp(x, dtype).get_output(0)


def batch_norm_infer(x: Tensor, running_mean: Tensor, running_var: Tensor, epsilon=1e-5, axis=1) -> Tensor:
    assert len(x.shape) == 4 and axis == 1
    assert len(running_mean.shape) == 1 and len(running_var.shape) == 1
    assert x.shape[1] == running_mean.shape[0] == running_var.shape[0]
    running_mean = running_mean.unsqueeze([0, 2, 3])  # [1, c, 1, 1]
    running_var = running_var.unsqueeze([0, 2, 3])
    return (x - running_mean) * (running_var + epsilon).rsqrt()


def take(data: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    return TakeOp(data, indices, axis).get_output(0)


raise DeprecationWarning('Should not use this module.')

