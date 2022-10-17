from __future__ import annotations
from typing import Union, Sequence, Tuple, Optional, List

from hidet.ir.node import Node
from hidet.ir.type import ScalarType, TensorType, tensor_type
from hidet.ir.expr import Expr, convert, Var, var, And, if_then_else
from .reduce_operations import ReduceOperation


class ComputeNode(Expr):
    def __init__(self, name):
        self.name: Optional[str] = name


class ScalarNode(ComputeNode):
    def __init__(self, name, data_type, reduce_compute=None):
        super().__init__(name)
        self.data_type: ScalarType = data_type
        self.scalar_compute: Optional[ScalarCompute] = reduce_compute

    def is_input(self) -> bool:
        return self.scalar_compute is None


class TensorNode(ComputeNode):
    def __init__(self, name, data_type, tensor_compute=None):
        super().__init__(name)
        self.data_type: TensorType = data_type
        self.tensor_compute: Optional[TensorCompute] = tensor_compute

    def astext(self):
        from hidet.ir.functors.printer import IRPrinter
        printer = IRPrinter()
        doc = printer.print_tensor_nodes([self])
        return str(doc.trim())

    def is_input(self) -> bool:
        return self.tensor_compute is None

    def const_shape(self) -> List[int]:
        return self.data_type.const_shape()

    def protect_read(self, indices, default_value=0.0) -> Expr:
        conds = []
        assert len(indices) == len(self.data_type.shape)
        for index, extent in zip(indices, self.data_type.shape):
            conds.append(0 <= index)
            conds.append(index < extent)
        return if_then_else(And.join(*conds), self.__getitem__(indices), default_value)


class ComputePrimitive(Node):
    pass


class ScalarCompute(ComputePrimitive):
    pass


class TensorCompute(ComputePrimitive):
    def as_grid_compute(self) -> GridCompute:
        if not isinstance(self, GridCompute):
            raise TypeError("Current object is not a grid compute.")
        return self


class GridCompute(TensorCompute):
    def __init__(
            self,
            input_tensors,
            input_scalars,
            shape: Sequence[Expr],
            axes: Sequence[Var],
            value: Expr
    ):
        self.input_tensors: List[TensorNode] = list(input_tensors)
        self.input_scalars: List[ScalarNode] = list(input_scalars)
        self.shape: Tuple[Expr] = tuple(shape)
        self.axes: Tuple[Var] = tuple(axes)
        self.value: Expr = value

        assert all(isinstance(v, TensorNode) for v in self.input_tensors) and all(isinstance(v, ScalarNode) for v in self.input_scalars)


class ReduceCompute(ScalarCompute):
    def __init__(
            self,
            input_tensors,
            input_scalars,
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

        assert all(isinstance(v, TensorNode) for v in self.input_tensors) and all(isinstance(v, ScalarNode) for v in self.input_scalars)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


class ArgReduceCompute(ScalarCompute):
    def __init__(
            self,
            input_tensors,
            input_scalars,
            extent: Expr,
            axis: Var,
            value: Expr,
            reduce_operation: ReduceOperation,
            index_dtype: ScalarType
    ):
        self.input_tensors: List[TensorNode] = list(input_tensors)
        self.input_scalars: List[ScalarNode] = list(input_scalars)
        self.extent: Expr = extent
        self.axis: Var = axis
        self.value: Expr = value
        self.reduce_operation: ReduceOperation = reduce_operation
        self.index_dtype: ScalarType = index_dtype

        assert all(isinstance(v, TensorNode) for v in self.input_tensors) and all(isinstance(v, ScalarNode) for v in self.input_scalars)


def scalar_input(name, dtype):
    if isinstance(dtype, str):
        dtype = ScalarType(dtype)
    else:
        assert isinstance(dtype, ScalarType)
    return ScalarNode(name, dtype, reduce_compute=None)


def tensor_input(name, base_type, shape, layout=None):
    data_type = tensor_type(base_type, shape, layout)
    return TensorNode(name, data_type, tensor_compute=None)


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


def compute(name, shape, fcompute, layout=None) -> TensorNode:
    from hidet.ir.functors import infer_type, simplify, collect
    shape = [convert(v) for v in shape]
    axes = [var() for _ in shape]
    value = simplify(convert(fcompute(*axes)))
    return TensorNode(
        name=name,
        data_type=tensor_type(dtype=infer_type(value), shape=shape, layout=layout),
        tensor_compute=GridCompute(
            input_tensors=collect(value, TensorNode, stop_when_found=True),
            input_scalars=collect(value, ScalarNode, stop_when_found=True),
            shape=shape,
            axes=axes,
            value=value
        )
    )


def arg_reduce(extent: Union[int, Expr], fcompute, reduce_type: str, index_dtype: str = 'int32') -> ScalarNode:
    from hidet.ir.functors import collect, simplify
    extent = convert(extent)
    axis = var()
    value = simplify(convert(fcompute(axis)))
    return ScalarNode(
        name='arg_{}'.format(reduce_type),
        data_type=ScalarType(index_dtype),
        reduce_compute=ArgReduceCompute(
            input_tensors=collect(value, TensorNode, stop_when_found=True),
            input_scalars=collect(value, ScalarNode, stop_when_found=True),
            extent=extent,
            axis=axis,
            value=value,
            reduce_operation=ReduceOperation.from_name(reduce_type),
            index_dtype=ScalarType(index_dtype)
        )
    )


