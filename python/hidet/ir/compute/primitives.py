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
from hidet.ir.expr import Expr, convert, Var, Constant, var
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

    @property
    def const_shape(self) -> List[int]:
        return list(int(v) for v in self.type.shape)

    @property
    def shape(self) -> List[Expr]:
        return list(self.type.shape)

    def is_concrete(self) -> bool:
        return all(isinstance(v, Constant) for v in self.shape)

    def is_symbolic(self) -> bool:
        return not self.is_concrete()


class ScalarInput(ScalarNode):
    def __init__(self, name: str, dtype: DataType):
        super().__init__(name)
        self.dtype: DataType = dtype


class TensorInput(TensorNode):
    def __init__(self, name: str, ttype: TensorType):
        super().__init__(name)
        self.ttype: TensorType = ttype


class ReduceCompute(ScalarNode):
    def __init__(
        self,
        name,
        shape: Sequence[Expr],
        axes: Sequence[Var],
        value: Expr,
        reduce_operation: ReduceOperation,
        accumulate_dtype: DataType,
    ):
        super().__init__(name)
        self.shape: Tuple[Expr] = tuple(shape)
        self.axes: Tuple[Var] = tuple(axes)
        self.value: Expr = value
        self.reduce_operation: ReduceOperation = reduce_operation
        self.accumulate_dtype: DataType = accumulate_dtype


class ArgReduceCompute(ScalarNode):
    def __init__(
        self, name, extent: Expr, axis: Var, value: Expr, reduce_operation: ReduceOperation, index_dtype: DataType
    ):
        super().__init__(name)
        self.extent: Expr = extent
        self.axis: Var = axis
        self.value: Expr = value
        self.reduce_operation: ReduceOperation = reduce_operation
        self.index_dtype: DataType = index_dtype


class GridCompute(TensorNode):
    def __init__(
        self, name: str, shape: Sequence[Expr], axes: Sequence[Var], value: Expr, layout: Optional[DataLayout] = None
    ):
        super().__init__(name)
        self._shape: Tuple[Expr] = tuple(shape)
        self.axes: Tuple[Var] = tuple(axes)
        self.value: Expr = value
        self.layout: Optional[DataLayout] = layout


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
    from hidet.ir.tools import simplify

    reduce_type = ReduceType(reduce_type)
    shape = [convert(v) for v in shape]
    axes = [var() for _ in shape]
    value = simplify(convert(fcompute(*axes)))
    if name is None:
        name = f'acc_{reduce_type.name}'
    return ReduceCompute(
        name=name,
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
    from hidet.ir.tools import simplify

    shape = [convert(v) for v in shape]
    axes = [var() for _ in shape]
    value = simplify(convert(fcompute(*axes)))
    return GridCompute(name=name, shape=shape, axes=axes, value=value, layout=layout)


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
    from hidet.ir.tools import simplify  # pylint: disable=import-outside-toplevel

    reduce_type = ReduceType(reduce_type)
    extent = convert(extent)
    axis = var()
    value = simplify(convert(fcompute(axis)))
    if name is None:
        name = f'arg_{reduce_type.name}'
    return ArgReduceCompute(
        name=name,
        extent=extent,
        axis=axis,
        value=value,
        reduce_operation=ReduceOperation.from_name(reduce_type),
        index_dtype=data_type(index_dtype),
    )
