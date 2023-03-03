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
# pylint: disable=redefined-builtin, unnecessary-lambda
from typing import List, Callable, Any, Union, Optional, Dict

from hidet.ir import primitives
from hidet.ir import expr, dtypes
from hidet.ir.type import DataType
from hidet.ir.expr import Constant, if_then_else
from hidet.utils import prod
from .utils import Task, Operator, Tensor, TensorNode, InverseMap, compute, input_like
from .utils import broadcast_shape, broadcast_shapes, broadcast_indices


class UnaryElementwiseTask(Task):
    def __init__(self, name: str, x: TensorNode, op: Callable[[Any], Any]):
        shape = x.const_shape()
        y = compute(name='y', shape=shape, fcompute=lambda *indices: op(x.__getitem__(indices)))
        super().__init__(
            name=name,
            inputs=[x],
            outputs=[y],
            inverse_map={x: InverseMap.from_lambda(lambda *indices: list(indices), num_args=len(x.ttype.shape))},
        )


class BinaryElementwiseTask(Task):
    def __init__(self, name: str, x: TensorNode, y: TensorNode, op: Callable[[Any, Any], Any]):
        x_shape = x.const_shape()
        y_shape = y.const_shape()
        z_shape = broadcast_shape(x_shape, y_shape)

        z = compute(
            name='z',
            shape=z_shape,
            fcompute=lambda *indices: op(
                x[broadcast_indices(indices, x_shape, z_shape)], y[broadcast_indices(indices, y_shape, z_shape)]
            ),
        )

        inverse_map = {}
        for inp, inp_shape in zip([x, y], [x_shape, y_shape]):
            if prod(inp_shape) == prod(z_shape):
                inverse_map[inp] = InverseMap.from_lambda(
                    lambda *indices: [0 for _ in range(len(z_shape) - len(inp_shape))] + list(indices),
                    num_args=len(inp_shape),
                )

        super().__init__(name=name, inputs=[x, y], outputs=[z], inverse_map=inverse_map)


class VariadicElementwiseTask(Task):
    def __init__(self, name: str, args: List[TensorNode], op: Callable[[Any], Any]):
        shapes = [arg.const_shape() for arg in args]
        out_shape = broadcast_shapes(shapes)
        out = compute(
            name='out',
            shape=out_shape,
            fcompute=lambda *indices: op(
                *[arg[broadcast_indices(indices, shape, out_shape)] for shape, arg in zip(shapes, args)]
            ),
        )
        super().__init__(
            name=name,
            inputs=list(args),
            outputs=[out],
            inverse_map={
                v: InverseMap.identity(len(v_shape))
                for v, v_shape in zip(args, shapes)
                if prod(v_shape) == prod(out_shape)
            },
        )


class WhereTask(Task):
    def __init__(self, cond: TensorNode, x: TensorNode, y: TensorNode):
        cond_shape = cond.const_shape()
        x_shape = x.const_shape()
        y_shape = y.const_shape()
        z_shape = broadcast_shape(cond_shape, broadcast_shape(x_shape, y_shape))

        z = compute(
            name='z',
            shape=z_shape,
            fcompute=lambda *indices: expr.if_then_else(
                cond=cond[broadcast_indices(indices, cond_shape, z_shape)],
                then_expr=x[broadcast_indices(indices, x_shape, z_shape)],
                else_expr=y[broadcast_indices(indices, y_shape, z_shape)],
            ),
        )

        super().__init__(
            name='where',
            inputs=[cond, x, y],
            outputs=[z],
            inverse_map={
                v: InverseMap.identity(len(v_shape))
                for v, v_shape in zip([cond, x, y], [cond_shape, x_shape, y_shape])
                if prod(v_shape) == prod(z_shape)
            },
        )


class UnaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, op, name: str, attributes: Optional[Dict[str, Any]] = None):
        super().__init__(inputs=[x], task=UnaryElementwiseTask(name, input_like(x, 'x'), op=op), attributes=attributes)


class BinaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, y: Tensor, op, name: str):
        super().__init__(inputs=[x, y], task=BinaryElementwiseTask(name, input_like(x, 'x'), input_like(y, 'y'), op=op))


def resolve_dtype(tensor_dtype: DataType, scalar_dtype: DataType) -> DataType:
    if tensor_dtype.is_integer() and scalar_dtype.is_float():
        return scalar_dtype
    else:
        return tensor_dtype


class AddScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Constant):
        dtype = resolve_dtype(x.dtype, scalar.type)
        super().__init__(x, op=lambda v: v + dtype(scalar), attributes={'scalar': scalar}, name='adds')


class SubScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Constant):
        dtype = resolve_dtype(x.dtype, scalar.type)
        super().__init__(x, op=lambda v: v - dtype(scalar), attributes={'scalar': scalar}, name='subs')


class RSubScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Constant):
        dtype = resolve_dtype(x.dtype, scalar.type)
        super().__init__(x, op=lambda v: dtype(scalar) - v, attributes={'scalar': scalar}, name='rsubs')


class MultiplyScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Constant):
        dtype = resolve_dtype(x.dtype, scalar.type)
        super().__init__(x, op=lambda v: v * dtype(scalar), attributes={'scalar': scalar}, name='muls')


class DivideScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Constant):
        dtype = resolve_dtype(x.dtype, scalar.type)
        super().__init__(x, op=lambda v: v / dtype(scalar), attributes={'scalar': scalar}, name='divs')


class RDivideScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Constant):
        dtype = resolve_dtype(x.dtype, scalar.type)
        super().__init__(x, op=lambda v: dtype(scalar) / v, attributes={'scalar': scalar}, name='rdivs')


class SqrtOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.sqrt(v), name='sqrt')


class ErfOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.erf(v), name='erf')


class ExpOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.exp(v), name='exp')


class Expm1Op(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.expm1(v), name='expm1')


class LogOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.log(v), name='log')


class Log2Op(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.log2(v), name='log2')


class Log10Op(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.log10(v), name='log10')


class Log1pOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.log1p(v), name='log1p')


class RsqrtOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: primitives.rsqrt(v), name='rsqrt')


class PowOp(BinaryElementwiseOp):
    def __init__(self, x, y):
        super().__init__(x, y, op=lambda x, y: primitives.pow(x, y), name='pow')


class NegativeOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: -v, name='negative')


class ReciprocalOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: x.dtype.one / v, name='reciprocal')


class AddOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a + b, name='add')


class SubtractOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a - b, name='subtract')


class MultiplyOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a * b, name='mul')


class DivideOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a / b, name='div')


class SinOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.sin(a), name='sin')


class CosOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.cos(a), name='cos')


class TanOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.tan(a), name='tan')


class SinhOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.sinh(a), name='sinh')


class CoshOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.cosh(a), name='cosh')


class TanhOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.tanh(a), name='tanh')


class AcosOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.acos(a), name='acos')


class AsinOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.asin(a), name='asin')


class AtanOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.atan(a), name='atan')


class Atan2Op(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: primitives.atan2(a, b), name='atan2')


class AcoshOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.acosh(a), name='acosh')


class AsinhOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.asinh(a), name='asinh')


class AtanhOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.atanh(a), name='atanh')


class SquareOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: a * a, name='square')


class CubeOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: a * a * a, name='cube')


class AbsOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: if_then_else(a >= x.dtype.zero, a, -a), name='abs')


class FloorOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.floor(a), name='floor')


class RoundOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.round(a), name='round')


class TruncOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.trunc(a), name='trunc')


class CeilOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.ceil(a), name='ceil')


class IsFiniteOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.isfinite(a), name='isfinite')


class IsInfOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.isinf(a), name='isinf')


class IsNanOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: primitives.isnan(a), name='isnan')


class SignOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(
            x,
            op=lambda a: if_then_else(
                a > x.dtype.zero, x.dtype.one, if_then_else(a < x.dtype.zero, -x.dtype.one, x.dtype.zero)
            ),
            name='sign',
        )


class RightShiftOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: expr.RightShift(a, b), name='rightshift')


class LeftShiftOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: expr.LeftShift(a, b), name='leftshift')


class BitwiseAndOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a & b, name='bitwise_and')


class BitwiseNotOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda a: expr.BitwiseNot(a), name='bitwise_not')


class BitwiseOrOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a | b, name='bitwise_or')


class BitwiseXorOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: a ^ b, name='bitwise_xor')


class ModOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, op=lambda a, b: primitives.mod(a, b), name='mod')


class WhereOp(Operator):
    def __init__(self, cond: Tensor, x: Tensor, y: Tensor):
        super().__init__(
            inputs=[cond, x, y],
            task=WhereTask(input_like(cond, 'cond'), input_like(x, 'x'), input_like(y, 'y')),
            name='where',
        )


class MaxOp(Operator):
    def __init__(self, *tensors: Tensor):
        def scalar_max(args: List[expr.Expr]):
            if len(args) == 1:
                return args[0]
            else:
                return primitives.max(args[0], scalar_max(args[1:]))

        super().__init__(
            inputs=list(tensors),
            task=VariadicElementwiseTask(
                name='max',
                args=[input_like(x, f'x{idx}') for idx, x in enumerate(tensors)],
                op=lambda *args: scalar_max(args),
            ),
            name='max',
        )


class MinOp(Operator):
    def __init__(self, *tensors: Tensor):
        def scalar_min(args: List[expr.Expr]):
            if len(args) == 1:
                return args[0]
            else:
                return primitives.min(args[0], scalar_min(args[1:]))

        super().__init__(
            inputs=list(tensors),
            task=VariadicElementwiseTask(
                name='min',
                args=[input_like(x, f'x{idx}') for idx, x in enumerate(tensors)],
                op=lambda *args: scalar_min(args),
            ),
            name='min',
        )


def binary_arithmetic(
    x: Union[Tensor, Constant, float, int],
    y: Union[Tensor, Constant, float, int],
    tensor_scalar_op: Callable[[Tensor, Constant], Tensor],
    scalar_tensor_op: Callable[[Constant, Tensor], Tensor],
    tensor_tensor_op: Callable[[Tensor, Tensor], Tensor],
) -> Union[Tensor, float, int]:
    if not (isinstance(x, (Tensor, float, int, Constant)) and isinstance(y, (Tensor, float, int, Constant))):
        raise ValueError(
            'Only support add/sub/mul/div between hidet.Tensor, float, int, and Constant. got {} and {}'.format(
                type(x), type(y)
            )
        )
    if not isinstance(x, Tensor) and not isinstance(y, Tensor):
        raise ValueError('One of x and y must be a Tensor')

    if isinstance(x, Tensor) and isinstance(y, Tensor) and len(x.shape) == len(y.shape) == 0:
        return tensor_tensor_op(x, y)

    if isinstance(x, int):
        x = dtypes.int32(x)
    elif isinstance(x, float):
        x = dtypes.float32(x)
    elif isinstance(x, Tensor) and len(x.shape) == 0:
        if x.trace is None and x.storage is not None:
            x = x.dtype(x.item())

    if isinstance(y, int):
        y = dtypes.int32(y)
    elif isinstance(y, float):
        y = dtypes.float32(y)
    elif isinstance(y, Tensor) and len(y.shape) == 0:
        if y.trace is None and y.storage is not None:
            y = y.dtype(y.item())

    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return tensor_tensor_op(x, y)
    elif isinstance(x, Tensor):
        return tensor_scalar_op(x, y)
    elif isinstance(y, Tensor):
        return scalar_tensor_op(x, y)
    else:
        assert False


def add(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
    return binary_arithmetic(
        x,
        y,
        lambda a, b: AddScalarOp(a, b).get_output(0),
        lambda a, b: AddScalarOp(b, a).get_output(0),
        lambda a, b: AddOp(a, b).get_output(0),
    )


def subtract(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
    return binary_arithmetic(
        x,
        y,
        lambda a, b: SubScalarOp(a, b).get_output(0),
        lambda a, b: RSubScalarOp(b, a).get_output(0),
        lambda a, b: SubtractOp(a, b).get_output(0),
    )


def multiply(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
    return binary_arithmetic(
        x,
        y,
        lambda a, b: MultiplyScalarOp(a, b).get_output(0),
        lambda a, b: MultiplyScalarOp(b, a).get_output(0),
        lambda a, b: MultiplyOp(a, b).get_output(0),
    )


def divide(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
    return binary_arithmetic(
        x,
        y,
        lambda a, b: DivideScalarOp(a, b).get_output(0),
        lambda a, b: RDivideScalarOp(b, a).get_output(0),
        lambda a, b: DivideOp(a, b).get_output(0),
    )


def sqrt(x: Tensor) -> Tensor:
    return SqrtOp(x).get_output(0)


def pow(x: Tensor, y: Tensor) -> Tensor:
    return PowOp(x, y).get_output(0)


def erf(x: Tensor) -> Tensor:
    return ErfOp(x).get_output(0)


def exp(x: Tensor) -> Tensor:
    return ExpOp(x).get_output(0)


def expm1(x: Tensor) -> Tensor:
    return Expm1Op(x).get_output(0)


def log(x: Tensor) -> Tensor:
    return LogOp(x).get_output(0)


def log2(x: Tensor) -> Tensor:
    return Log2Op(x).get_output(0)


def log10(x: Tensor) -> Tensor:
    return Log10Op(x).get_output(0)


def log1p(x: Tensor) -> Tensor:
    return Log1pOp(x).get_output(0)


def rsqrt(x: Tensor) -> Tensor:
    return RsqrtOp(x).get_output(0)


def negative(x: Tensor) -> Tensor:
    return NegativeOp(x).get_output(0)


def positive(x: Tensor) -> Tensor:
    return x


def reciprocal(x: Tensor) -> Tensor:
    return ReciprocalOp(x).get_output(0)


def sin(x: Tensor) -> Tensor:
    return SinOp(x).get_output(0)


def cos(x: Tensor) -> Tensor:
    return CosOp(x).get_output(0)


def tan(x: Tensor) -> Tensor:
    return TanOp(x).get_output(0)


def asin(x: Tensor) -> Tensor:
    return AsinOp(x).get_output(0)


def acos(x: Tensor) -> Tensor:
    return AcosOp(x).get_output(0)


def atan(x: Tensor) -> Tensor:
    return AtanOp(x).get_output(0)


def atan2(x: Tensor, y: Tensor) -> Tensor:
    return Atan2Op(x, y).get_output(0)


def sinh(x: Tensor) -> Tensor:
    return SinhOp(x).get_output(0)


def cosh(x: Tensor) -> Tensor:
    return CoshOp(x).get_output(0)


def tanh(x: Tensor) -> Tensor:
    return TanhOp(x).get_output(0)


def asinh(x: Tensor) -> Tensor:
    return AsinhOp(x).get_output(0)


def acosh(x: Tensor) -> Tensor:
    return AcoshOp(x).get_output(0)


def atanh(x: Tensor) -> Tensor:
    return AtanhOp(x).get_output(0)


def square(x: Tensor) -> Tensor:
    return SquareOp(x).get_output(0)


def cube(x: Tensor) -> Tensor:
    return CubeOp(x).get_output(0)


def isfinite(x: Tensor) -> Tensor:
    return IsFiniteOp(x).get_output(0)


def isinf(x: Tensor) -> Tensor:
    return IsInfOp(x).get_output(0)


def isnan(x: Tensor) -> Tensor:
    return IsNanOp(x).get_output(0)


def sign(x: Tensor) -> Tensor:
    return SignOp(x).get_output(0)


def where(cond: Tensor, x: Tensor, y: Tensor) -> Tensor:
    if cond.dtype != dtypes.boolean:
        raise ValueError('The condition tensor must have dtype "bool", but got {}'.format(cond.dtype.name))
    return WhereOp(cond, x, y).get_output(0)


def maximum(a: Tensor, b: Tensor, *others: Tensor) -> Tensor:
    args = [a, b] + list(others)
    return MaxOp(*args).get_output(0)


def minimum(a: Tensor, b: Tensor, *others: Tensor) -> Tensor:
    args = [a, b] + list(others)
    return MinOp(*args).get_output(0)


def mod(x: Tensor, y: Tensor) -> Tensor:
    return ModOp(x, y).get_output(0)


remainder = mod


def abs(x: Tensor) -> Tensor:
    return AbsOp(x).get_output(0)


def bitwise_right_shift(x: Tensor, y: Tensor) -> Tensor:
    return RightShiftOp(x, y).get_output(0)


def bitwise_left_shift(x: Tensor, y: Tensor) -> Tensor:
    return LeftShiftOp(x, y).get_output(0)


def bitwise_and(x: Tensor, y: Tensor) -> Tensor:
    return BitwiseAndOp(x, y).get_output(0)


def bitwise_invert(x: Tensor) -> Tensor:
    return BitwiseNotOp(x).get_output(0)


def bitwise_or(x: Tensor, y: Tensor) -> Tensor:
    return BitwiseOrOp(x, y).get_output(0)


def bitwise_xor(x: Tensor, y: Tensor) -> Tensor:
    return BitwiseXorOp(x, y).get_output(0)


def floor(x: Tensor) -> Tensor:
    return FloorOp(x).get_output(0)


def ceil(x: Tensor) -> Tensor:
    return CeilOp(x).get_output(0)


def round(x: Tensor) -> Tensor:
    return RoundOp(x).get_output(0)


def trunc(x: Tensor) -> Tensor:
    return TruncOp(x).get_output(0)


def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    max_val = maximum(x, y)
    return log(exp(x - max_val) + exp(y - max_val)) + max_val
