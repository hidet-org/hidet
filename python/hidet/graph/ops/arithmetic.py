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
from typing import List, Callable, Any, Union, Optional, Dict, Sequence

from hidet.ir import primitives
from hidet.ir import Var, expr, dtypes
from hidet.ir.type import DataType
from hidet.ir.expr import Expr, if_then_else, logical_or, is_constant, is_true
from hidet.ir.tools import rewrite
from hidet.utils import prod, same_list
from .utils import Task, Operator, Tensor, TensorNode, InverseMap, compute, input_like
from .utils import broadcast_shape, broadcast_shapes, broadcast_indices
from .utils import normalize_slice, normalize_dim

PyScalar = Union[int, float, bool]


# In order for the subgraph rewrite of Composite Elementwise Operator to work,
# we need to store the callable in an Operator object. But lambda cannot be pickled,
# so we define auxiliary classes UnaryElementwiseOperation and BinaryElementwiseOperation
# below which can be saved to disk and can act as a callable to construct the task.
class UnaryElementwiseOperation:
    def __init__(self, x, y, name, attributes, task_attributes):
        self.x: Var = x
        self.y: Expr = y
        self.attributes = attributes
        self.task_attributes = task_attributes
        self.name: str = name

    def __call__(self, x):
        return rewrite(self.y, {self.x: x})

    @staticmethod
    def from_callable(op: Callable[[Any], Any], name, attributes, task_attributes):
        x = expr.var('x')
        y = op(x)
        return UnaryElementwiseOperation(x, y, name, attributes, task_attributes)


class BinaryElementwiseOperation:
    def __init__(self, left_operand, right_operand, out, name):
        self.left_operand: Var = left_operand
        self.right_operand: Var = right_operand
        self.out: Expr = out
        self.name: str = name

    def __call__(self, left_operand, right_operand):
        return rewrite(self.out, {self.left_operand: left_operand, self.right_operand: right_operand})

    @staticmethod
    def from_callable(op: Callable[[Any, Any], Any], name):
        left_operand = expr.var('left_operand')
        right_operand = expr.var('right_operand')
        out = op(left_operand, right_operand)
        return BinaryElementwiseOperation(left_operand, right_operand, out, name)


class UnaryElementwiseTask(Task):
    def __init__(self, name: str, x: TensorNode, op: Callable[[Any], Any], attrs=None):
        shape = x.shape
        y = compute(name='y', shape=shape, fcompute=lambda *indices: op(x.__getitem__(indices)))
        super().__init__(
            name=name,
            inputs=[x],
            outputs=[y],
            inverse_map={x: InverseMap.from_lambda(lambda *indices: list(indices), num_args=len(x.type.shape))},
            attributes={} if attrs is None else attrs,
        )


class BinaryElementwiseTask(Task):
    def __init__(self, name: str, x: TensorNode, y: TensorNode, op: Callable[[Any, Any], Any]):
        z_shape = broadcast_shape(x.shape, y.shape)

        z = compute(
            name='z',
            shape=z_shape,
            fcompute=lambda *indices: op(
                x[broadcast_indices(indices, x.shape, z_shape)], y[broadcast_indices(indices, y.shape, z_shape)]
            ),
        )

        inverse_map = {}
        for inp, inp_shape in zip([x, y], [x.shape, y.shape]):
            if same_list(inp_shape, z_shape):
                inverse_map[inp] = InverseMap.from_lambda(lambda *indices: indices, num_args=len(inp_shape))
            elif prod(inp_shape) == prod(z_shape):
                inverse_map[inp] = InverseMap.from_lambda(
                    lambda *indices: [0 for _ in range(len(z_shape) - len(inp_shape))] + list(indices),
                    num_args=len(inp_shape),
                )

        super().__init__(name=name, inputs=[x, y], outputs=[z], inverse_map=inverse_map)


class VariadicElementwiseTask(Task):
    def __init__(self, name: str, args: List[TensorNode], op: Callable[[Any], Any]):
        shapes = [arg.shape for arg in args]
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
                if is_true(prod(v_shape) == prod(out_shape)) and len(v_shape) == len(out_shape)
            },
        )


class CompositeElementwiseTask(Task):
    def __init__(
        self,
        name: str,
        x: TensorNode,
        left_unary_op: Callable[[Any], Any],
        right_unary_op: Callable[[Any], Any],
        binary_op: Callable[[Any, Any], Any],
        attrs=None,
    ):
        def composite_op(binary_op, left_unary_op, right_unary_op, x):
            if left_unary_op is None:
                left_unary_op = lambda x: x
            if right_unary_op is None:
                right_unary_op = lambda x: x
            return binary_op(left_unary_op(x), right_unary_op(x))

        shape = x.shape

        z = compute(
            name='z',
            shape=shape,
            fcompute=lambda *indices: composite_op(binary_op, left_unary_op, right_unary_op, x.__getitem__(indices)),
        )

        super().__init__(
            name=name,
            inputs=[x],
            outputs=[z],
            inverse_map={x: InverseMap.from_lambda(lambda *indices: list(indices), num_args=len(x.type.shape))},
            attributes={} if attrs is None else attrs,
        )


class WhereTask(Task):
    def __init__(self, cond: TensorNode, x: TensorNode, y: TensorNode):
        cond_shape = cond.shape
        x_shape = x.shape
        y_shape = y.shape
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
                if prod(v_shape) == prod(z_shape) and len(v_shape) == len(z_shape)
            },
        )


class SetStridedSliceTask(Task):
    def __init__(
        self,
        data: TensorNode,
        starts: List[Optional[int]],
        ends: List[Optional[int]],
        axes: List[int],
        strides: List[int],
        setvalue: [Union[int, float]],
    ):
        assert len(starts) == len(ends) == len(axes) == len(strides)
        if len(axes) != len(set(axes)):
            raise ValueError('Duplicated axes in slice, axes: {}'.format(axes))
        output_shape = list(data.shape)
        axis2info = {}
        for axis, start, end, stride in zip(axes, starts, ends, strides):
            if stride == 0:
                raise NotImplementedError(
                    'Stride can not be 0 in slicing: '
                    'starts {} ends {} axes {} strides {}.'.format(starts, ends, axes, strides)
                )
            if is_constant(output_shape[axis]) and output_shape[axis] < 0:
                raise NotImplementedError(
                    'Slice result can not be: '
                    'starts {} ends {} axes {} strides {}'.format(starts, ends, axes, strides)
                )
            axis2info[axis] = (start, end, stride)

        def fmap(indices):
            ret = data.type.dtype(setvalue)
            for axis, index in enumerate(indices):
                start, end, stride = axis2info[axis]
                ret = if_then_else(
                    logical_or(index < start, index >= end, (index - start) % stride != 0), data[indices], ret
                )
            return ret

        out = compute('out', shape=output_shape, fcompute=lambda *indices: fmap(indices))
        super().__init__(name='set_slice', inputs=[data], outputs=[out])


class RollTask(Task):
    def __init__(self, x: TensorNode, shifts: Sequence[int], dims: Sequence[int]):
        output_shape = list(x.shape)

        def fmap(indices):
            data_indices = []
            for axis, index in enumerate(indices):
                if axis in dims:
                    i = dims.index(axis)
                    if shifts[i] > 0:
                        data_indices.append(
                            if_then_else(
                                index - shifts[i] >= 0, index - shifts[i], index + output_shape[axis] - shifts[i]
                            )
                        )
                    else:
                        data_indices.append(
                            if_then_else(
                                index - shifts[i] < output_shape[axis],
                                index - shifts[i],
                                index - output_shape[axis] - shifts[i],
                            )
                        )
                else:
                    data_indices.append(index)
            return x[data_indices]

        out = compute('out', shape=output_shape, fcompute=lambda *indices: fmap(indices))
        super().__init__(name='roll', inputs=[x], outputs=[out])


class UnaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, op, name: str, attributes: Optional[Dict[str, Any]] = None, task_attributes=None):
        if attributes is None:
            attributes = {}
        self.op = UnaryElementwiseOperation.from_callable(op, name, attributes, task_attributes)
        super().__init__(
            inputs=[x],
            attributes=attributes,
            task=UnaryElementwiseTask(name, input_like(x, 'x'), op=op, attrs=task_attributes),
        )


class BinaryElementwiseOp(Operator):
    def __init__(self, x: Tensor, y: Tensor, op, name: str):
        self.op = BinaryElementwiseOperation.from_callable(op, name)
        super().__init__(
            inputs=[x, y],
            attributes={},
            task=BinaryElementwiseTask(name, input_like(x, 'x'), input_like(y, 'y'), op=op),
        )


def get_dtype(scalar: Expr):
    from hidet.ir.tools import infer_type

    inferred_type = infer_type(scalar)
    if not isinstance(inferred_type, DataType):
        raise TypeError(f'Expected scalar to be of type DataType, got {type(inferred_type)}')
    return inferred_type


class CompositeElementwiseOp(Operator):
    def __init__(
        self,
        x: Tensor,
        left_unary_op: UnaryElementwiseOperation,
        right_unary_op: UnaryElementwiseOperation,
        binary_op: BinaryElementwiseOperation,
    ):
        name = 'composite'
        for op in [left_unary_op, right_unary_op, binary_op]:
            if op is not None:
                name += '_' + op.name
        attributes = {'left_unary_op': left_unary_op, 'right_unary_op': right_unary_op, 'binary_op': binary_op}
        super().__init__(
            inputs=[x],
            attributes=attributes,
            task=CompositeElementwiseTask(name, input_like(x, 'x'), left_unary_op, right_unary_op, binary_op),
        )


def resolve_dtype(tensor_dtype: DataType, scalar_dtype: DataType) -> DataType:
    if tensor_dtype.is_integer() and (scalar_dtype.is_float() or scalar_dtype.is_complex()):
        return scalar_dtype
    elif tensor_dtype.is_float() and scalar_dtype.is_complex():
        return scalar_dtype
    else:
        return tensor_dtype


class AddScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Expr):
        dtype = resolve_dtype(x.dtype, get_dtype(scalar))
        super().__init__(x, op=lambda v: v + dtype(scalar), attributes={'scalar': scalar}, name='adds')


class SubScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Expr):
        dtype = resolve_dtype(x.dtype, get_dtype(scalar))
        super().__init__(x, op=lambda v: v - dtype(scalar), attributes={'scalar': scalar}, name='subs')


class RSubScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Expr):
        dtype = resolve_dtype(x.dtype, get_dtype(scalar))
        super().__init__(x, op=lambda v: dtype(scalar) - v, attributes={'scalar': scalar}, name='rsubs')


class MultiplyScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Expr):
        dtype = resolve_dtype(x.dtype, get_dtype(scalar))
        super().__init__(x, op=lambda v: v * dtype(scalar), attributes={'scalar': scalar}, name='muls')


class DivideScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Expr):
        dtype = resolve_dtype(x.dtype, get_dtype(scalar))
        super().__init__(x, op=lambda v: v / dtype(scalar), attributes={'scalar': scalar}, name='divs')


class RDivideScalarOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, scalar: Expr):
        dtype = resolve_dtype(x.dtype, get_dtype(scalar))
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


class ClampOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, min_value: Union[int, float], max_value: Union[int, float]):
        assert isinstance(min_value, (int, float))
        assert isinstance(max_value, (int, float))
        min_value = x.dtype(min_value)
        max_value = x.dtype(max_value)
        super().__init__(
            x,
            op=lambda a: if_then_else(a < min_value, min_value, if_then_else(a > max_value, max_value, a)),
            name='clamp',
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
            attributes={},
            task=WhereTask(input_like(cond, 'cond'), input_like(x, 'x'), input_like(y, 'y')),
        )


class WhereScalarScalarOp(Operator):
    def __init__(self, cond: Tensor, x: PyScalar, y: PyScalar):
        if isinstance(x, int) and isinstance(y, int):
            dtype = dtypes.default_int_dtype
        elif isinstance(x, float) or isinstance(y, float):
            dtype = dtypes.default_float_dtype
        else:
            raise ValueError(f'Unsupported scalar type: {type(x)}')
        x, y = dtype(x), dtype(y)
        super().__init__(
            inputs=[cond],
            attributes={'x': x, 'y': y},
            task=UnaryElementwiseTask(name='where', x=input_like(cond, 'cond'), op=lambda a: if_then_else(a, x, y)),
        )


class WhereScalarTensorOp(Operator):
    def __init__(self, cond: Tensor, y: Tensor, x: PyScalar):
        dtype = y.dtype
        x = dtype(x)
        super().__init__(
            inputs=[cond, y],
            attributes={'x': x},
            task=BinaryElementwiseTask(
                name='where', x=input_like(cond, 'cond'), y=input_like(y, 'y'), op=lambda a, b: if_then_else(a, x, b)
            ),
        )


class WhereTensorScalarOp(Operator):
    def __init__(self, cond: Tensor, x: Tensor, y: PyScalar):
        y = x.dtype(y)
        super().__init__(
            inputs=[cond, x],
            attributes={'y': y},
            task=BinaryElementwiseTask(
                name='where', x=input_like(cond, 'cond'), y=input_like(x, 'x'), op=lambda a, b: if_then_else(a, b, y)
            ),
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
            attributes={},
            task=VariadicElementwiseTask(
                name='max',
                args=[input_like(x, f'x{idx}') for idx, x in enumerate(tensors)],
                op=lambda *args: scalar_max(args),
            ),
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
            attributes={},
            task=VariadicElementwiseTask(
                name='min',
                args=[input_like(x, f'x{idx}') for idx, x in enumerate(tensors)],
                op=lambda *args: scalar_min(args),
            ),
        )


class SetStridedSliceOp(Operator):
    def __init__(
        self,
        data: Tensor,
        starts: Sequence[Optional[int]],
        ends: Sequence[Optional[int]],
        strides: Optional[Sequence[Optional[int]]] = None,
        setvalue: Optional[Union[int, float]] = 0.0,
    ):
        starts, ends, axes, strides = normalize_slice(data.shape, starts, ends, axes=None, strides=strides)
        task = SetStridedSliceTask(input_like(data, 'data'), starts, ends, axes, strides, setvalue)
        super().__init__(
            inputs=[data],
            attributes={'starts': starts, 'ends': ends, 'strides': strides, 'setvalue': setvalue},
            task=task,
        )


class RollOp(Operator):
    def __init__(self, x: Tensor, shifts: Sequence[int], dims: Sequence[int]):
        if not len(shifts) == len(dims):
            raise ValueError('Roll must have same size shifts and dims, got {} and {}'.format(len(shifts), len(dims)))
        task = RollTask(input_like(x, 'x'), shifts, dims)
        super().__init__(inputs=[x], attributes={'shifts': shifts, 'dims': dims}, task=task)


Scalar = Union[Expr, float, int, complex]


def binary_arithmetic(
    x: Union[Tensor, Scalar],
    y: Union[Tensor, Scalar],
    tensor_scalar_op: Callable[[Tensor, Scalar], Tensor],
    scalar_tensor_op: Callable[[Scalar, Tensor], Tensor],
    tensor_tensor_op: Callable[[Tensor, Tensor], Tensor],
    scalar_scalar_op: Callable[[Scalar, Scalar], Scalar],
) -> Union[Tensor, float, int]:
    if not (isinstance(x, (Tensor, Expr, complex, float, int)) and isinstance(y, (Tensor, Expr, complex, float, int))):
        raise ValueError(
            'Only support add/sub/mul/div between hidet.Tensor, float, int, and Expr. got {} and {}'.format(
                type(x), type(y)
            )
        )

    def normalize_scalar(v):
        if isinstance(v, Expr):
            return v
        elif isinstance(v, bool):
            return dtypes.boolean(v)
        elif isinstance(v, int):
            return dtypes.int32(v)
        elif isinstance(v, float):
            return dtypes.float32(v)
        elif isinstance(v, complex):
            return dtypes.complex64(v)
        else:
            raise RuntimeError('Unsupported type {}'.format(type(v)))

    if isinstance(x, Tensor) and isinstance(y, Tensor):
        if x.device != y.device:
            # normalize to the same device
            if x.device.is_cpu() and len(x.shape) == 0:
                x = x.to(device=y.device)
                return binary_arithmetic(x, y, tensor_scalar_op, scalar_tensor_op, tensor_tensor_op, scalar_scalar_op)
            if y.device.is_cpu() and len(y.shape) == 0:
                y = y.to(device=x.device)
                return binary_arithmetic(x, y, tensor_scalar_op, scalar_tensor_op, tensor_tensor_op, scalar_scalar_op)
        # simplify the tensor vs tensor case where one tensor is a scalar
        if len(x.shape) == 0 and x.storage:
            x = x.dtype(x.item())
            return binary_arithmetic(x, y, tensor_scalar_op, scalar_tensor_op, tensor_tensor_op, scalar_scalar_op)
        if len(y.shape) == 0 and y.storage:
            y = y.dtype(y.item())
            return binary_arithmetic(x, y, tensor_scalar_op, scalar_tensor_op, tensor_tensor_op, scalar_scalar_op)
        return tensor_tensor_op(x, y)
    elif isinstance(x, Tensor):
        return tensor_scalar_op(x, normalize_scalar(y))
    elif isinstance(y, Tensor):
        return scalar_tensor_op(normalize_scalar(x), y)
    else:
        if isinstance(x, Expr) or isinstance(y, Expr):
            return scalar_scalar_op(normalize_scalar(x), normalize_scalar(y))
        else:
            return scalar_scalar_op(x, y)


def add(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
    return binary_arithmetic(
        x,
        y,
        lambda a, b: AddScalarOp(a, b).outputs[0],
        lambda a, b: AddScalarOp(b, a).outputs[0],
        lambda a, b: AddOp(a, b).outputs[0],
        lambda a, b: a + b,
    )


def subtract(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
    return binary_arithmetic(
        x,
        y,
        lambda a, b: SubScalarOp(a, b).outputs[0],
        lambda a, b: RSubScalarOp(b, a).outputs[0],
        lambda a, b: SubtractOp(a, b).outputs[0],
        lambda a, b: a - b,
    )


def multiply(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
    return binary_arithmetic(
        x,
        y,
        lambda a, b: MultiplyScalarOp(a, b).outputs[0],
        lambda a, b: MultiplyScalarOp(b, a).outputs[0],
        lambda a, b: MultiplyOp(a, b).outputs[0],
        lambda a, b: a * b,
    )


def divide(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
    return binary_arithmetic(
        x,
        y,
        lambda a, b: DivideScalarOp(a, b).outputs[0],
        lambda a, b: RDivideScalarOp(b, a).outputs[0],
        lambda a, b: DivideOp(a, b).outputs[0],
        lambda a, b: a / b,
    )


def sqrt(x: Tensor) -> Tensor:
    return SqrtOp(x).outputs[0]


def pow(x: Tensor, y: Tensor) -> Tensor:
    return PowOp(x, y).outputs[0]


def erf(x: Tensor) -> Tensor:
    return ErfOp(x).outputs[0]


def exp(x: Tensor) -> Tensor:
    return ExpOp(x).outputs[0]


def expm1(x: Tensor) -> Tensor:
    return Expm1Op(x).outputs[0]


def log(x: Tensor) -> Tensor:
    return LogOp(x).outputs[0]


def log2(x: Tensor) -> Tensor:
    return Log2Op(x).outputs[0]


def log10(x: Tensor) -> Tensor:
    return Log10Op(x).outputs[0]


def log1p(x: Tensor) -> Tensor:
    return Log1pOp(x).outputs[0]


def rsqrt(x: Tensor) -> Tensor:
    return RsqrtOp(x).outputs[0]


def negative(x: Tensor) -> Tensor:
    return NegativeOp(x).outputs[0]


def positive(x: Tensor) -> Tensor:
    return x


def reciprocal(x: Tensor) -> Tensor:
    return ReciprocalOp(x).outputs[0]


def sin(x: Tensor) -> Tensor:
    return SinOp(x).outputs[0]


def cos(x: Tensor) -> Tensor:
    return CosOp(x).outputs[0]


def tan(x: Tensor) -> Tensor:
    return TanOp(x).outputs[0]


def asin(x: Tensor) -> Tensor:
    return AsinOp(x).outputs[0]


def acos(x: Tensor) -> Tensor:
    return AcosOp(x).outputs[0]


def atan(x: Tensor) -> Tensor:
    return AtanOp(x).outputs[0]


def atan2(x: Tensor, y: Tensor) -> Tensor:
    return Atan2Op(x, y).outputs[0]


def sinh(x: Tensor) -> Tensor:
    return SinhOp(x).outputs[0]


def cosh(x: Tensor) -> Tensor:
    return CoshOp(x).outputs[0]


def tanh(x: Tensor) -> Tensor:
    return TanhOp(x).outputs[0]


def asinh(x: Tensor) -> Tensor:
    return AsinhOp(x).outputs[0]


def acosh(x: Tensor) -> Tensor:
    return AcoshOp(x).outputs[0]


def atanh(x: Tensor) -> Tensor:
    return AtanhOp(x).outputs[0]


def square(x: Tensor) -> Tensor:
    return SquareOp(x).outputs[0]


def cube(x: Tensor) -> Tensor:
    return CubeOp(x).outputs[0]


def isfinite(x: Tensor) -> Tensor:
    return IsFiniteOp(x).outputs[0]


def isinf(x: Tensor) -> Tensor:
    return IsInfOp(x).outputs[0]


def isnan(x: Tensor) -> Tensor:
    return IsNanOp(x).outputs[0]


def sign(x: Tensor) -> Tensor:
    return SignOp(x).outputs[0]


def clamp(x: Tensor, min: Union[Tensor, float, int], max: Union[Tensor, float, int]) -> Tensor:
    if isinstance(min, Tensor) or isinstance(max, Tensor):
        raise NotImplementedError('clamp with tensor min/max is not implemented yet')
    return ClampOp(x, min, max).outputs[0]


def where(cond: Tensor, x: Union[Tensor, PyScalar], y: Union[Tensor, PyScalar]) -> Tensor:
    if cond.dtype != dtypes.boolean:
        raise ValueError('The condition tensor must have dtype "bool", but got {}'.format(cond.dtype.name))
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return WhereOp(cond, x, y).outputs[0]
    elif isinstance(x, Tensor) and isinstance(y, (int, float, complex)):
        return WhereTensorScalarOp(cond, x=x, y=y).outputs[0]
    elif isinstance(x, (int, float, complex)) and isinstance(y, Tensor):
        return WhereScalarTensorOp(cond, x=x, y=y).outputs[0]
    elif isinstance(x, (int, float, complex)) and isinstance(y, (int, float, complex)):
        return WhereScalarScalarOp(cond, x=x, y=y).outputs[0]
    else:
        raise ValueError('Invalid arguments for where: x={}, y={}'.format(x, y))


def maximum(a: Tensor, b: Tensor, *others: Tensor) -> Tensor:
    args = [a, b] + list(others)
    return MaxOp(*args).outputs[0]


def minimum(a: Tensor, b: Tensor, *others: Tensor) -> Tensor:
    args = [a, b] + list(others)
    return MinOp(*args).outputs[0]


def mod(x: Tensor, y: Tensor) -> Tensor:
    return ModOp(x, y).outputs[0]


def remainder(x: Tensor, y: Tensor) -> Tensor:
    return mod(x, y)


def abs(x: Tensor) -> Tensor:
    return AbsOp(x).outputs[0]


def bitwise_right_shift(x: Tensor, y: Tensor) -> Tensor:
    return RightShiftOp(x, y).outputs[0]


def bitwise_left_shift(x: Tensor, y: Tensor) -> Tensor:
    return LeftShiftOp(x, y).outputs[0]


def bitwise_and(x: Tensor, y: Tensor) -> Tensor:
    return BitwiseAndOp(x, y).outputs[0]


def bitwise_invert(x: Tensor) -> Tensor:
    return BitwiseNotOp(x).outputs[0]


def bitwise_or(x: Tensor, y: Tensor) -> Tensor:
    return BitwiseOrOp(x, y).outputs[0]


def bitwise_xor(x: Tensor, y: Tensor) -> Tensor:
    return BitwiseXorOp(x, y).outputs[0]


def floor(x: Tensor) -> Tensor:
    return FloorOp(x).outputs[0]


def ceil(x: Tensor) -> Tensor:
    return CeilOp(x).outputs[0]


def round(x: Tensor) -> Tensor:
    return RoundOp(x).outputs[0]


def trunc(x: Tensor) -> Tensor:
    return TruncOp(x).outputs[0]


def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    max_val = maximum(x, y)
    return log(exp(x - max_val) + exp(y - max_val)) + max_val


def roll(x: Tensor, shifts: Union[int, Sequence[int]], dims: Union[int, Sequence[int]] = None) -> Tensor:
    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(dims, int):
        dims = [dims]
    if dims is None:
        from .transform import flatten, reshape

        shape = x.shape
        return reshape(RollOp(flatten(x), shifts, dims=[0]).outputs[0], shape)
    dims = normalize_dim(dims, len(x.shape))
    return RollOp(x, shifts, dims).outputs[0]


# out = binary_op(left_unary_op(x), right_unary_op(x)); This allows more fusion opportunity.
def composite_elementwise(
    x: Tensor,
    left_unary_op: UnaryElementwiseOperation,
    right_unary_op: UnaryElementwiseOperation,
    binary_op: BinaryElementwiseOperation,
) -> Tensor:
    return CompositeElementwiseOp(x, left_unary_op, right_unary_op, binary_op).outputs[0]


def set_strided_slice(
    data: Tensor,
    starts: Sequence[Optional[int]],
    ends: Sequence[Optional[int]],
    strides: Optional[Sequence[Optional[int]]] = None,
    setvalue: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    return SetStridedSliceOp(data, starts, ends, strides, setvalue).outputs[0]
