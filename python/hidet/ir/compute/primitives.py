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
# pylint: disable=import-outside-toplevel
from __future__ import annotations
from typing import Union, Sequence, Tuple, Optional, List
from hidet.ir.type import DataType, TensorType, tensor_type, data_type
from hidet.ir.expr import Expr, convert, Var, var
from hidet.ir.layout import DataLayout
from .reduce_operations import ReduceOperation, ReduceType


class ComputeNode(Expr):
    def __init__(self, name):
        self.name: str = name


class ScalarNode(ComputeNode):
    def __init__(self, name: str):
        super().__init__(name)
        self._type: Optional[DataType] = None

    @property
    def type(self) -> DataType:
        if self._type is None:
            from hidet.ir.tools import infer_type

            self._type = infer_type(self)
        return self._type


class TensorNode(ComputeNode):
    def __init__(self, name: str):
        super().__init__(name)
        self._type: Optional[TensorType] = None

    @property
    def type(self) -> TensorType:
        if self._type is None:
            from hidet.ir.tools import infer_type

            self._type = infer_type(self)
        return self._type

    @property
    def ndim(self) -> int:
        return len(self.type.shape)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.type.shape]


class ScalarInput(ScalarNode):
    def __init__(self, name: str, dtype: DataType):
        super().__init__(name)
        self.dtype: DataType = dtype


class TensorInput(TensorNode):
    def __init__(self, name: str, ttype: TensorType):
        super().__init__(name)
        self.ttype: TensorType = ttype

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.ttype.shape]

    @property
    def shape(self) -> Tuple[Expr]:
        return self.ttype.shape


class ReduceCompute(ScalarNode):
    def __init__(
        self,
        name,
        input_tensors: Sequence[TensorNode],
        input_scalars: Sequence[ScalarNode],
        shape: Sequence[Expr],
        axes: Sequence[Var],
        value: Expr,
        reduce_operation: ReduceOperation,
        accumulate_dtype: DataType,
    ):
        super().__init__(name)
        self.input_tensors: Tuple[TensorNode] = tuple(input_tensors)
        self.input_scalars: Tuple[ScalarNode] = tuple(input_scalars)
        self.shape: Tuple[Expr] = tuple(shape)
        self.axes: Tuple[Var] = tuple(axes)
        self.value: Expr = value
        self.reduce_operation: ReduceOperation = reduce_operation
        self.accumulate_dtype: DataType = accumulate_dtype

        assert all(isinstance(v, TensorNode) for v in self.input_tensors)
        assert all(isinstance(v, ScalarNode) for v in self.input_scalars)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


class ArgReduceCompute(ScalarNode):
    def __init__(
        self,
        name,
        input_tensors: Sequence[TensorNode],
        input_scalars: Sequence[ScalarNode],
        extent: Expr,
        axis: Var,
        value: Expr,
        reduce_operation: ReduceOperation,
        index_dtype: DataType,
    ):
        super().__init__(name)
        self.input_tensors: Tuple[TensorNode] = tuple(input_tensors)
        self.input_scalars: Tuple[ScalarNode] = tuple(input_scalars)
        self.extent: Expr = extent
        self.axis: Var = axis
        self.value: Expr = value
        self.reduce_operation: ReduceOperation = reduce_operation
        self.index_dtype: DataType = index_dtype

        assert all(isinstance(v, TensorNode) for v in self.input_tensors)
        assert all(isinstance(v, ScalarNode) for v in self.input_scalars)


class GridCompute(TensorNode):
    def __init__(
        self,
        name: str,
        input_tensors: Sequence[TensorNode],
        input_scalars: Sequence[ScalarNode],
        shape: Sequence[Expr],
        axes: Sequence[Var],
        value: Expr,
        layout: Optional[DataLayout] = None,
    ):
        super().__init__(name)
        self.input_tensors: Tuple[TensorNode] = tuple(input_tensors)
        self.input_scalars: Tuple[ScalarNode] = tuple(input_scalars)
        self.shape: Tuple[Expr] = tuple(shape)
        self.axes: Tuple[Var] = tuple(axes)
        self.value: Expr = value
        self.layout: Optional[DataLayout] = layout

        assert all(isinstance(v, TensorNode) for v in self.input_tensors)
        assert all(isinstance(v, ScalarNode) for v in self.input_scalars)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


# class ScalarNode(ComputeNode):
#     def __init__(self, name, dtype=None, reduce_compute=None):
#         super().__init__(name)
#         self.dtype: Optional[DataType] = dtype
#         self.scalar_compute: Optional[ScalarCompute] = reduce_compute
#
#     def type(self) -> DataType:
#         if self.dtype is None:
#             from hidet.ir.tools import infer_type
#             self.dtype = infer_type(self.scalar_compute)
#         return self.dtype
#
#
# class TensorNode(ComputeNode):
#     """
#     A node in the compute DAG that represents a tensor.
#
#     Can be either an input or an intermediate node. The input node is created by
#     :py:func:`~hidet.ir.compute.tensor_input`. and the intermediate node is created by
#     :py:func:`~hidet.ir.compute.compute`.
#     """
#
#     def __init__(self, name, ttype=None, tensor_compute=None):
#         super().__init__(name)
#         self.ttype: Optional[TensorType] = ttype
#         self.tensor_compute: Optional[TensorCompute] = tensor_compute
#
#     def astext(self):
#         from hidet.ir.tools.printer import IRPrinter  # pylint: disable=import-outside-toplevel
#
#         printer = IRPrinter()
#         doc = printer.print_tensor_nodes([self])
#         return str(doc.trim())
#
#     @property
#     def ndim(self) -> int:
#         return len(self.ttype.shape)
#
#     def type(self) -> TensorType:
#         if self.ttype is None:
#             from hidet.ir.tools import infer_type
#             self.ttype = infer_type(self.tensor_compute)
#         return self.ttype
#
#     def is_input(self) -> bool:
#         return self.tensor_compute is None
#
#     def const_shape(self) -> List[int]:
#         return self.ttype.const_shape()
#
#     def protect_read(self, indices, default_value=0.0) -> Expr:
#         conds = []
#         assert len(indices) == len(self.ttype.shape)
#         for index, extent in zip(indices, self.ttype.shape):
#             conds.append(0 <= index)
#             conds.append(index < extent)
#         return if_then_else(LogicalAnd.join(*conds), self[indices], default_value)
#
#
# class ComputePrimitive(Node):
#     pass
#
#
# class ScalarCompute(ComputePrimitive):
#     pass
#
#
# class TensorCompute(ComputePrimitive):
#     def as_grid_compute(self) -> GridCompute:
#         if not isinstance(self, GridCompute):
#             raise TypeError("Current object is not a grid compute.")
#         return self
#
#
# class GridCompute(TensorCompute):
#     def __init__(self, input_tensors, input_scalars, shape: Sequence[Expr], axes: Sequence[Var], value: Expr):
#         self.input_tensors: List[TensorNode] = list(input_tensors)
#         self.input_scalars: List[ScalarNode] = list(input_scalars)
#         self.shape: Tuple[Expr] = tuple(shape)
#         self.axes: Tuple[Var] = tuple(axes)
#         self.value: Expr = value
#
#         assert all(isinstance(v, TensorNode) for v in self.input_tensors)
#         assert all(isinstance(v, ScalarNode) for v in self.input_scalars)
#
#
# class ReduceCompute(ScalarCompute):
#     def __init__(
#         self,
#         input_tensors,
#         input_scalars,
#         shape: Sequence[Expr],
#         axes: Sequence[Var],
#         value: Expr,
#         reduce_operation: ReduceOperation,
#         accumulate_dtype: DataType,
#     ):
#         self.input_tensors: List[TensorNode] = list(input_tensors)
#         self.input_scalars: List[ScalarNode] = list(input_scalars)
#         self.shape: Tuple[Expr] = tuple(shape)
#         self.axes: Tuple[Var] = tuple(axes)
#         self.value: Expr = value
#         self.reduce_operation: ReduceOperation = reduce_operation
#         self.accumulate_dtype: DataType = accumulate_dtype
#
#         assert all(isinstance(v, TensorNode) for v in self.input_tensors)
#         assert all(isinstance(v, ScalarNode) for v in self.input_scalars)
#
#     def const_shape(self) -> List[int]:
#         return [int(v) for v in self.shape]
#
#
# class ArgReduceCompute(ScalarCompute):
#     def __init__(
#         self,
#         input_tensors,
#         input_scalars,
#         extent: Expr,
#         axis: Var,
#         value: Expr,
#         reduce_operation: ReduceOperation,
#         index_dtype: DataType,
#     ):
#         self.input_tensors: List[TensorNode] = list(input_tensors)
#         self.input_scalars: List[ScalarNode] = list(input_scalars)
#         self.extent: Expr = extent
#         self.axis: Var = axis
#         self.value: Expr = value
#         self.reduce_operation: ReduceOperation = reduce_operation
#         self.index_dtype: DataType = index_dtype
#
#         assert all(isinstance(v, TensorNode) for v in self.input_tensors)
#         assert all(isinstance(v, ScalarNode) for v in self.input_scalars)


def scalar_input(name, dtype) -> ScalarInput:
    """
    Define an input scalar node.

    Parameters
    ----------
    name: str
        The name of the input scalar.
    dtype: str or DataType
        The scalar type of the input scalar.

    Returns
    -------
    ret: ScalarInput
        The input scalar node.
    """
    if isinstance(dtype, str):
        dtype = data_type(dtype)
    else:
        assert isinstance(dtype, DataType)
    return ScalarInput(name, dtype)


def tensor_input(name, dtype, shape, layout=None) -> TensorInput:
    """
    Define an input tensor node.

    Parameters
    ----------
    name: str
        The name of the input tensor.
    dtype: str or DataType
        The scalar type of the tensor.
    shape: Sequence[Expr or int]
        The shape of the tensor.
    layout: DataLayout, optional
        The layout of the tensor.

    Returns
    -------
    ret: TensorInput
        The input tensor node.
    """
    ttype = tensor_type(dtype, shape, layout)
    return TensorInput(name, ttype)


def reduce(shape, fcompute, reduce_type, accumulate_dtype='float32', name: Optional[str] = None) -> ReduceCompute:
    """
    Define a reduction node.

    Parameters
    ----------
    shape: Sequence[int or Expr]
        The domain of the reduction.
    fcompute: Callable[[Sequence[Var]], Expr]
        The compute function. It takes a list of reduction variables and returns the reduction value.
    reduce_type: ReduceType or str
        The type of the reduction.
    accumulate_dtype: str or DataType
        The data type of the accumulator.
    name: Optional[str]
        The name hint for the output. If not specified, the name will be generated automatically.

    Returns
    -------
    ret: ReduceCompute
        The reduction node.
    """
    from hidet.ir.tools import simplify, collect  # pylint: disable=import-outside-toplevel

    reduce_type = ReduceType(reduce_type)
    shape = [convert(v) for v in shape]
    axes = [var() for _ in shape]
    value = simplify(convert(fcompute(*axes)))
    if name is None:
        name = f'acc_{reduce_type.name}'
    return ReduceCompute(
        name=name,
        input_tensors=collect(value, TensorNode, stop_when_found=True),
        input_scalars=collect(value, ScalarNode, stop_when_found=True),
        shape=shape,
        axes=axes,
        value=value,
        reduce_operation=ReduceOperation.from_name(reduce_type),
        accumulate_dtype=data_type(accumulate_dtype),
    )


def compute(name, shape: Sequence[Union[int, Expr]], fcompute, layout=None) -> TensorNode:
    """
    Define a grid compute node.

    Parameters
    ----------
    name: str
        The name of the compute node.
    shape: Sequence[Union[int, Expr]]
        The shape of the compute node.
    fcompute: Callable[[Sequence[Var]], Expr]
        The compute function. It takes a list of index variables and returns the output value corresponding to the
        index.
    layout: DataLayout, optional
        The layout of the compute node.

    Returns
    -------
    ret: TensorNode
        The grid compute node.
    """
    from hidet.ir.tools import simplify, collect  # pylint: disable=import-outside-toplevel

    shape = [convert(v) for v in shape]
    axes = [var() for _ in shape]
    value = simplify(convert(fcompute(*axes)))
    return GridCompute(
        name=name,
        input_tensors=collect(value, TensorNode, stop_when_found=True),
        input_scalars=collect(value, ScalarNode, stop_when_found=True),
        shape=shape,
        axes=axes,
        value=value,
        layout=layout,
    )


def arg_reduce(extent, fcompute, reduce_type, index_dtype='int64', name=None) -> ScalarNode:
    """
    Define an arg reduction node.

    Parameters
    ----------
    extent: int or Expr
        The domain of the reduction.
    fcompute: Callable[[Var], Expr]
        The compute function. It takes a reduction variable and returns the value to compare.
    reduce_type: str or ReduceType
        The type of the reduction.
    index_dtype: str or DataType
        The data type of the index.
    name: str, optional
        The name of the output. If not specified, the name will be generated automatically.

    Returns
    -------
    ret: ScalarNode
        The arg reduction node.
    """
    from hidet.ir.tools import collect, simplify  # pylint: disable=import-outside-toplevel

    reduce_type = ReduceType(reduce_type)
    extent = convert(extent)
    axis = var()
    value = simplify(convert(fcompute(axis)))
    if name is None:
        name = f'arg_{reduce_type.name}'
    return ArgReduceCompute(
        name=name,
        input_tensors=collect(value, TensorNode, stop_when_found=True),
        input_scalars=collect(value, ScalarNode, stop_when_found=True),
        extent=extent,
        axis=axis,
        value=value,
        reduce_operation=ReduceOperation.from_name(reduce_type),
        index_dtype=data_type(index_dtype),
    )
