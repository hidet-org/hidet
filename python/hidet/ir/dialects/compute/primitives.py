from typing import Union, Sequence, Tuple, Optional, List, Dict, Any
from hidet.ir.type import ScalarType, TensorType, Scope, tensor_type, scalar_type
from hidet.ir.expr import Expr, Constant, convert, Var, var, And, if_then_else
from hidet.utils.info import float_type_min_value
from .reduce_operations import ReduceOperation


class ComputeNode(Expr):
    def __init__(self, name):
        self.name: Optional[str] = name


class ScalarNode(ComputeNode):
    def __init__(self, name, data_type, reduce_compute=None):
        super().__init__(name)
        self.data_type: ScalarType = data_type
        self.reduce_compute: Optional[ReduceCompute] = reduce_compute

    def is_input(self) -> bool:
        return self.reduce_compute is None


class TensorNode(ComputeNode):
    def __init__(self, name, data_type, grid_compute=None):
        super().__init__(name)
        self.data_type: TensorType = data_type
        self.grid_compute: Optional[GridCompute] = grid_compute

    def is_input(self) -> bool:
        return self.grid_compute is None

    def const_shape(self) -> List[int]:
        return self.data_type.const_shape()

    def protect_read(self, indices, default_value=0.0) -> Expr:
        conds = []
        assert len(indices) == len(self.data_type.shape)
        for index, extent in zip(indices, self.data_type.shape):
            conds.append(0 <= index)
            conds.append(index < extent)
        return if_then_else(And.join(*conds), self.__getitem__(indices), default_value)


class GridCompute:
    def __init__(
            self,
            input_tensors: Sequence[TensorNode],
            input_scalars: Sequence[ScalarNode],
            shape: Sequence[Expr],
            axes: Sequence[Var],
            value: Expr
    ):
        self.input_tensors: List[TensorNode] = list(input_tensors)
        self.input_scalars: List[ScalarNode] = list(input_scalars)
        self.shape: Tuple[Expr] = tuple(shape)
        self.axes: Tuple[Var] = tuple(axes)
        self.value: Expr = value


class ReduceCompute:
    def __init__(
            self,
            input_tensors: Sequence[TensorNode],
            input_scalars: Sequence[ScalarNode],
            shape: Sequence[Expr],
            axes: Sequence[Var],
            value: Expr,
            reduce_operation: ReduceOperation,
            accumulate_dtype: ScalarType
    ):
        self.input_tensors: List[TensorNode] = list(input_tensors)
        self.input_scalars: List[ScalarNode] = list(input_scalars)
        self.shape: Tuple[Expr] = tuple(shape)
        self.axes: Tuple[Var] = tuple(axes)
        self.value: Expr = value
        self.reduce_operation: ReduceOperation = reduce_operation
        self.accumulate_dtype: ScalarType = accumulate_dtype

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


def scalar_input(name, dtype):
    if isinstance(dtype, str):
        dtype = ScalarType(dtype)
    else:
        assert isinstance(dtype, ScalarType)
    return ScalarNode(name, dtype, reduce_compute=None)


def tensor_input(name, base_type, shape, scope=None, layout=None):
    data_type = tensor_type(scope, base_type, shape, layout)
    return TensorNode(name, data_type, grid_compute=None)


def reduce(shape: Sequence[Union[int, Expr]], fcompute, reduce_type: str, accumulate_dtype: str = 'float32') -> ScalarNode:
    from hidet.ir.functors import infer_type, simplify, collect
    shape = [convert(v) for v in shape]
    axes = [var() for _ in shape]
    value = simplify(convert(fcompute(*axes)))
    return ScalarNode(
        name=f'acc_{reduce_type}',
        data_type=infer_type(value),
        reduce_compute=ReduceCompute(
            input_tensors=collect(value, TensorNode, stop_when_found=True),
            input_scalars=collect(value, ScalarNode, stop_when_found=True),
            shape=shape,
            axes=axes,
            value=value,
            reduce_operation=ReduceOperation.from_name(reduce_type),
            accumulate_dtype=ScalarType(accumulate_dtype)
        )
    )


def compute(name, shape, fcompute, scope=None, layout=None) -> TensorNode:
    from hidet.ir.functors import infer_type, simplify, collect
    shape = [convert(v) for v in shape]
    axes = [var() for _ in shape]
    value = simplify(convert(fcompute(*axes)))
    return TensorNode(
        name=name,
        data_type=tensor_type('global', dtype=infer_type(value), shape=shape, layout=layout),
        grid_compute=GridCompute(
            input_tensors=collect(value, TensorNode, stop_when_found=True),
            input_scalars=collect(value, ScalarNode, stop_when_found=True),
            shape=shape,
            axes=axes,
            value=value
        )
    )

