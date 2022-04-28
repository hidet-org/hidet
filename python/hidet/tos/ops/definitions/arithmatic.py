from typing import List, Callable, Any

from hidet.ir import primitives
from .utils import Task, Operator, Tensor, TensorNode, compute, input_like


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


class UnaryElementwiseTask(Task):
    def __init__(self, name: str, x: TensorNode, op: Callable[[Any], Any]):
        shape = x.const_shape()
        y = compute(
            name='y',
            shape=shape,
            fcompute=lambda *indices: op(x.__getitem__(indices)),
            scope='global'
        )
        super().__init__(
            name=name,
            inputs=[x],
            outputs=[y]
        )


class BinaryElementwiseTask(Task):
    def __init__(self, name: str, x: TensorNode, y: TensorNode, op: Callable[[Any, Any], Any]):
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
        super().__init__(
            name=name,
            inputs=[x, y],
            outputs=[z]
        )


class UnaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, op, name: str):
        super().__init__(
            inputs=[x],
            task=UnaryElementwiseTask(name, input_like(x, 'x'), op=op)
        )


class BinaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, y: Tensor, op, name: str):
        super().__init__(
            inputs=[x, y],
            task=BinaryElementwiseTask(name, input_like(x, 'x'), input_like(y, 'y'), op=op)
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


class SinOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.cuda_sin(a), name='sin')


class CosOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.cuda_cos(a), name='cos')


class SquareOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: a * a, name='square')


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


def sin(x: Tensor) -> Tensor:
    return SinOp(x).get_output(0)


def cos(x: Tensor) -> Tensor:
    return CosOp(x).get_output(0)


def square(x: Tensor) -> Tensor:
    return SquareOp(x).get_output(0)
