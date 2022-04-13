from typing import List, Callable, Any

# from hidet.ir.primitives import cuda_sqrt, cuda_rsqrt
from hidet.ir import primitives
from ..common import Task, Operator, Tensor, TensorInput, Grid, compute, input_like


def broadcast_shape(x_shape: List[int], y_shape: List[int]) -> List[int]:
    """
    Broadcast two shapes with the same rule as numpy.
    Please refer to https://numpy.org/doc/stable/user/basics.broadcasting.html for details.
    """
    x_shape = [int(v) for v in x_shape]
    y_shape = [int(v) for v in y_shape]
    orig_shapes = x_shape, y_shape
    while len(x_shape) < len(y_shape):
        x_shape = [1] + x_shape
    while len(y_shape) < len(x_shape):
        y_shape = [1] + y_shape
    result_shape = []
    for p, q in zip(x_shape, y_shape):
        if p != q and p != 1 and q != 1:
            raise ValueError('can not broadcast two arrays with shape {} and {}'.format(orig_shapes[0], orig_shapes[1]))
        result_shape.append(max(p, q))
    return result_shape


def unary_elementwise_task(name, x: TensorInput, op: Callable[[Any], Any]):
    shape = x.const_shape()
    y = compute(
        name='y',
        shape=shape,
        fcompute=lambda *indices: op(x.__getitem__(indices)),
        scope='global'
    )
    return Task(
        name=name,
        computation=y,
        params=[x, y],
        worker=Grid()
    )


def binary_elementwise_task(name, x: TensorInput, y: TensorInput, op: Callable[[Any, Any], Any]):
    x_shape = x.const_shape()
    y_shape = y.const_shape()
    z_shape = broadcast_shape(x_shape, y_shape)

    def imap(indices, shape):
        # used to support broadcast
        pad_dim = len(z_shape) - len(shape)
        indices = list(indices[pad_dim:])
        for idx, dim in enumerate(shape):
            if int(dim) == 1:
                indices[idx] = 0
        return indices

    z = compute(
        name='z',
        shape=z_shape,
        fcompute=lambda *indices: op(x[imap(indices, x_shape)], y[imap(indices, y_shape)]),
        scope='global'
    )
    return Task(
        name=name,
        computation=z,
        params=[x, y, z],
        worker=Grid()
    )


class UnaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, op, name: str):
        super().__init__(
            inputs=[x],
            task=unary_elementwise_task(name, input_like(x, 'x'), op=op)
        )


class BinaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, y: Tensor, op, name: str):
        super().__init__(
            inputs=[x, y],
            task=binary_elementwise_task(name, input_like(x, 'x'), input_like(y, 'y'), op=op)
        )


class SqrtOp(UnaryElementwiseOp):
    def __init__(self, x):
        # todo: use a target-agnostic primitive function to define task
        #       and use a pass lower these functions into target-specific version.
        super().__init__(x, op=lambda v: primitives.cuda_sqrt(v), name='sqrt')


class ErfOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.cuda_erf(v), name='erf')


class TanhOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.cuda_tanh(v), name='erf')


class RsqrtOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.cuda_rsqrt(v), name='rsqrt')


class PowOp(BinaryElementwiseOp):
    def __init__(self, x, y):
        super().__init__(x, y, op=lambda x, y: primitives.cuda_pow(x, y), name='pow')


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


def tanh(x: Tensor) -> Tensor:
    return TanhOp(x).get_output(0)


def pow(x: Tensor, y: Tensor) -> Tensor:
    return PowOp(x, y).get_output(0)


def erf(x: Tensor) -> Tensor:
    return ErfOp(x).get_output(0)


def rsqrt(x: Tensor) -> Tensor:
    return RsqrtOp(x).get_output(0)


def neg(x: Tensor) -> Tensor:
    return NegOp(x).get_output(0)

