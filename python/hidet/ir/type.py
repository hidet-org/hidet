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
from typing import Sequence, Optional, Union, List, Tuple, Callable, Any

from hidet.ir.node import Node

# typing forward declaration
Expr = 'Expr'
Int = Union['Expr', int]


class TypeNode(Node):
    def __invert__(self) -> TypeNode:
        # get the pointer type that points to current type
        if isinstance(self, TensorType):
            return TensorPointerType.from_tensor_type(self)
        elif isinstance(self, DataType):
            return PointerType(base_type=self)
        elif isinstance(self, (PointerType, TensorPointerType)):
            return PointerType(base_type=self)
        else:
            raise ValueError('Can not recognize type {}'.format(self))


class DataType(TypeNode):
    """
    The data type that defines how to interpret the data in memory.
    """

    def __init__(self, name: str, short_name: str, nbytes: int):
        self._name = name
        self._short_name = short_name
        self._nbytes = nbytes

    def __str__(self):
        return 'hidet.{}'.format(self.name)

    def __eq__(self, other):
        return isinstance(other, DataType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __call__(self, value: Any):
        """
        Create a constant of current data type, or convert an existing Expr to current data type with cast expression.

        Parameters
        ----------
        value: Union[int, float, bool, list, tuple, Constant, Expr]
            The value of the constant or the value to be casted.

        Returns
        -------
        ret: Constant or Cast
            The constant or cast expression.
        """
        from hidet.ir import expr

        if isinstance(value, (int, float, bool, list, tuple)):
            return self.constant(value)
        elif isinstance(value, expr.Constant):
            return self.constant(value.value)
        elif isinstance(value, expr.Expr):
            return expr.cast(value, self)
        else:
            raise ValueError('Can not convert {} to {}'.format(value, self))

    def __getitem__(self, item):
        if not isinstance(item, (tuple, list)):
            item = (item,)
        return tensor_type(dtype=self, shape=list(item))

    @property
    def name(self) -> str:
        return self._name

    @property
    def short_name(self) -> str:
        return self._short_name

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def is_float(self) -> bool:
        raise NotImplementedError()

    def is_integer(self) -> bool:
        raise NotImplementedError()

    def is_vector(self) -> bool:
        raise NotImplementedError()

    def constant(self, value: Any):
        raise NotImplementedError()

    @property
    def one(self):
        raise NotImplementedError()

    @property
    def zero(self):
        raise NotImplementedError()

    @property
    def min_value(self):
        raise NotImplementedError()

    @property
    def max_value(self):
        raise NotImplementedError()


class TensorType(TypeNode):
    def __init__(self, dtype=None, shape=None, layout=None):
        """
        A tensor type.

        Parameters
        ----------
        dtype: DataType
            The data type of the tensor.
        shape: Tuple[Expr, ...]
            The shape of the tensor.
        layout: hidet.ir.layout.DataLayout
            The layout of the tensor.
        """
        from hidet.ir.layout import DataLayout

        self.dtype: DataType = dtype
        self.shape: Tuple[Expr] = shape
        self.layout: DataLayout = layout

    def __invert__(self):
        return TensorPointerType.from_tensor_type(self)

    def storage_bytes(self) -> Expr:
        return self.layout.size * self.dtype.nbytes

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


class VoidType(TypeNode):
    pass


class PointerType(TypeNode):
    def __init__(self, base_type, specifiers: Optional[Sequence[str]] = None, use_bracket: bool = False):
        super().__init__()
        if isinstance(base_type, str):
            base_type = data_type(base_type)
        self.base_type: TypeNode = base_type
        # todo: move the following attributes to DeclareStmt
        self.specifiers: List[str] = list(specifiers) if specifiers else []
        self.use_bracket: bool = use_bracket


class ReferenceType(TypeNode):
    def __init__(self, base_type):
        super().__init__()
        self.base_type = base_type


class TensorPointerType(TypeNode):
    def __init__(self, dtype, shape, layout):
        """
        A pointer type that points to tensor.

        Parameters
        ----------
        dtype: DataType
            The data type of the tensor.
        shape: Tuple[Expr, ...]
            The shape of the tensor.
        layout: hidet.ir.layout.DataLayout
            The layout of the tensor.
        """
        self.tensor_type: TensorType = tensor_type(dtype, shape, layout)

    @staticmethod
    def from_tensor_type(tp: TensorType) -> TensorPointerType:
        tpt = object.__new__(TensorPointerType)
        tpt.tensor_type = tp
        return tpt


TypeLike = Union[str, TypeNode]


class FuncType(TypeNode):
    def __init__(
        self,
        param_types: Optional[List[TypeLike]] = None,
        ret_type: Optional[TypeLike] = None,
        type_infer_func: Optional[Callable] = None,  # Callable[[a number of TypeNode], TypeNode]
    ):
        self.param_types = [self._convert_type(tp) for tp in param_types] if param_types is not None else None
        self.ret_type = self._convert_type(ret_type) if ret_type is not None else None
        self.type_infer_func = type_infer_func
        msg = 'Please provide either a static type or a type infer func'
        assert not all(v is None for v in [ret_type, type_infer_func]), msg

    def ret_type_on(self, arg_types: List[TypeNode]) -> TypeNode:
        if self.ret_type is not None:
            # todo: add type checking
            return self.ret_type
        else:
            return self.type_infer_func(arg_types)

    def _convert_type(self, tp: Union[str, TypeNode]):
        if isinstance(tp, str):
            return data_type(tp)
        else:
            return tp

    @staticmethod
    def from_func(func):
        return FuncType([param.type for param in func.params], func.ret_type)


def tensor_type(dtype, shape: Optional[Sequence[Union[int, Expr]]] = None, layout=None):
    """
    Construct a tensor type.

    One of shape and layout must be given.

    Parameters
    ----------
    dtype: str or DataType
        The scalar type of this tensor.

    shape: Sequence[Union[int, Expr]] or none
        The shape of the tensor. If not given, the shape in layout will be used.

    layout: hidet.ir.layout.DataLayout or none
        The layout of the tensor. If not given, the row major layout of given shape will
        be used.

    Returns
    -------
    ret: TensorType
        The constructed tensor type
    """
    from hidet.ir.expr import convert
    from hidet.ir.layout import DataLayout

    if isinstance(dtype, str):
        dtype = data_type(dtype)
    if not isinstance(dtype, DataType):
        raise ValueError('Scalar type expect a "str" or "ScalarType", but got {}'.format(type(dtype)))
    if shape is None and layout is None:
        raise ValueError('Tensor type must give either shape or layout')
    elif shape is None:
        assert isinstance(layout, DataLayout)
        shape = layout.shape
    elif layout is None:
        layout = DataLayout.row_major(list(shape))
    else:
        assert isinstance(layout, DataLayout)
        assert isinstance(shape, (list, tuple))
        for a, b in zip(shape, layout.shape):
            if int(a) != int(b):
                raise ValueError(
                    'The shape of tensor and the shape of layout are not compatible, '
                    '{} vs {}'.format(list(shape), list(layout.shape))
                )
    shape = convert(shape)
    return TensorType(dtype, shape, layout)


def pointer_type(base_type):
    return PointerType(base_type)


def tensor_pointer_type(dtype, shape=None, layout=None):
    ttype = tensor_type(dtype, shape, layout)
    return TensorPointerType(ttype.dtype, ttype.shape, ttype.layout)


def void_pointer():
    return PointerType(VoidType())


def data_type(dtype: Union[str, DataType]) -> DataType:
    from hidet.ir.dtypes import name2dtype, sname2dtype

    if isinstance(dtype, DataType):
        return dtype
    elif isinstance(dtype, str):
        if dtype in name2dtype:
            return name2dtype[dtype]
        elif dtype in sname2dtype:
            return sname2dtype[dtype]
        else:
            raise ValueError('Unknown data type: {}, candidates:\n{}'.format(dtype, '\n'.join(name2dtype.keys())))
    else:
        raise ValueError('Expect a string or a DataType, but got {}'.format(type(dtype)))


void_p = PointerType(VoidType())
