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
Int = Union[int, Expr]


class BaseType(Node):
    def __invert__(self) -> BaseType:
        # get the pointer type that points to current type
        if isinstance(self, TensorType):
            return TensorPointerType.from_tensor_type(self)
        elif isinstance(self, DataType):
            return PointerType(base_type=self)
        elif isinstance(self, (PointerType, TensorPointerType)):
            return PointerType(base_type=self)
        else:
            raise ValueError('Can not recognize type {}'.format(self))

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            if len(item) == 1:
                item = item[0]
            else:
                raise ValueError('Currently, only support 1-d array, but got {}'.format(item))
        return array_type(self, int(item))

    def is_void(self):
        return isinstance(self, VoidType)

    def is_tensor(self):
        return isinstance(self, TensorType)

    def is_pointer(self):
        return isinstance(self, (PointerType, TensorPointerType))

    def is_data_type(self):
        return isinstance(self, DataType)

    def is_func_type(self):
        return isinstance(self, FuncType)

    def is_string_type(self):
        return isinstance(self, StringType)

    def as_data_type(self) -> Optional[DataType]:
        if not isinstance(self, DataType):
            return None
        return self


class DataType(BaseType):
    """
    The data type that defines how to interpret the data in memory.

    """

    def __init__(self, name: str, short_name: str, nbytes: int):
        self._name: str = name
        self._short_name: str = short_name
        self._nbytes: int = nbytes

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

        built_types = (int, float, bool, complex)

        if (
            isinstance(value, built_types)
            or isinstance(value, (list, tuple))
            and all(isinstance(v, built_types) for v in value)
        ):
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

    @property
    def nbits(self) -> int:
        """
        Get the bit length of the data type

        Note:
        1. The bit length of the data type itself other than the bit length of its storage.
        2. For regular data types, the nbits can be computed from its nbytes property.
        3. For subbyte data types, the nbits is defined when constructing the data type,
        and this method will also be overridden for subbyte data types.
        4. In addition, we cannot access the nbytes for a subbyte data type, otherwise
        a type error will be raised.
        """
        return self._nbytes * 8

    @property
    def storage(self) -> DataType:
        """
        Get the actual storage type of the data type

        Note:
        1. The storage of a regular data type is the data type itself, while the storage
        of a subbyte type is the type of its actual storage. e.g., the storage of int4b is uint8
        2. The property will be overridden in the subclass of subbyte types.
        """
        return self

    def is_integer_subbyte(self) -> bool:
        raise NotImplementedError()

    def is_float(self) -> bool:
        raise NotImplementedError()

    def is_integer(self) -> bool:
        raise NotImplementedError()

    def is_complex(self) -> bool:
        raise NotImplementedError()

    def is_vector(self) -> bool:
        raise NotImplementedError()

    def is_boolean(self) -> bool:
        raise NotImplementedError()

    def is_any_float16(self) -> bool:
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


class TensorType(BaseType):
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
        self.shape: Tuple[Expr, ...] = shape
        self.layout: DataLayout = layout

    def __invert__(self):
        return TensorPointerType.from_tensor_type(self)

    def storage_bytes(self) -> Expr:
        if self.dtype.is_integer_subbyte():
            return self.layout.size * self.dtype.nbits // 8
        else:
            return self.layout.size * self.dtype.nbytes

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


class VoidType(BaseType):
    pass


class StringType(BaseType):
    pass


class PointerType(BaseType):
    def __init__(self, base_type, specifiers: Optional[Sequence[str]] = None, use_bracket: bool = False):
        super().__init__()
        if isinstance(base_type, str):
            base_type = data_type(base_type)
        self.base_type: BaseType = base_type
        # todo: move the following attributes to DeclareStmt
        self.specifiers: List[str] = list(specifiers) if specifiers else []
        self.use_bracket: bool = use_bracket

    def __call__(self, x):
        from hidet.ir.expr import Constant, Expr, constant, cast  # pylint: disable=redefined-outer-name

        if isinstance(x, int):
            return constant(x, self)
        elif isinstance(x, Constant):
            return constant(x.value, self)
        elif isinstance(x, Expr):
            return cast(x, self)
        else:
            raise ValueError('Can not convert {} to {}'.format(x, self))


class ReferenceType(BaseType):
    def __init__(self, base_type):
        super().__init__()
        self.base_type = base_type


class TensorPointerType(BaseType):
    def __init__(self, ttype: TensorType):
        """
        A pointer type that points to tensor.
        """
        self.tensor_type: TensorType = ttype

    @staticmethod
    def from_tensor_type(tp: TensorType) -> TensorPointerType:
        tpt = object.__new__(TensorPointerType)
        tpt.tensor_type = tp
        return tpt


class ArrayType(BaseType):
    def __init__(self, base_type, size: int):
        super().__init__()
        self.base_type: BaseType = base_type
        self.size: int = size

        assert isinstance(base_type, BaseType) and not isinstance(base_type, (ArrayType, TensorType))
        assert isinstance(size, int) and size >= 0


TypeLike = Union[str, BaseType]


class FuncType(BaseType):
    def __init__(
        self,
        param_types: Optional[List[TypeLike]] = None,
        ret_type: Optional[TypeLike] = None,
        type_infer_func: Optional[Callable] = None,  # Callable[[a number of BaseType], BaseType]
    ):
        self.param_types: Optional[List[BaseType]] = (
            [self._convert_type(tp) for tp in param_types] if param_types is not None else None
        )
        self.ret_type: Optional[BaseType] = self._convert_type(ret_type) if ret_type is not None else None
        self.type_infer_func: Optional[Callable[[List[BaseType]], BaseType]] = type_infer_func
        msg = 'Please provide either a static type or a type infer func'
        assert not all(v is None for v in [ret_type, type_infer_func]), msg

    def ret_type_on(self, arg_types: List[BaseType]) -> BaseType:
        if self.ret_type is not None:
            # todo: add type checking
            assert isinstance(self.ret_type, BaseType)
            return self.ret_type
        else:
            return self.type_infer_func(arg_types)

    def _convert_type(self, tp: Union[str, BaseType]):
        if isinstance(tp, str):
            return data_type(tp)
        else:
            return tp

    @staticmethod
    def from_func(func):
        return FuncType([param.type for param in func.params], func.ret_type)


class OpaqueType(BaseType):
    def __init__(self, cpp_name: str, *modifiers: str):
        self.cpp_name: str = cpp_name
        self.modifiers: Sequence[str] = modifiers


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
    from hidet.ir.layout import DataLayout, row_major

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
        layout = row_major(*shape)
    else:
        assert isinstance(layout, DataLayout)
        assert isinstance(shape, (list, tuple))
        assert len(shape) == len(layout.shape)
    shape = convert(shape)
    return TensorType(dtype, shape, layout)


def array_type(base_type: BaseType, size: int):
    return ArrayType(base_type, size)


def pointer_type(base_type):
    return PointerType(base_type)


def tensor_pointer_type(dtype, shape=None, layout=None):
    return TensorPointerType(tensor_type(dtype, shape, layout))


def string_type():
    return StringType()


def func_type(param_types, ret_type) -> FuncType:
    return FuncType(param_types, ret_type)


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


def type_equal(lhs: BaseType, rhs: BaseType) -> bool:
    """
    Check whether the two types are equal or not.

    Parameters
    ----------
    lhs: BaseType
        The first type to compare.
    rhs: BaseType
        The second type to compare.

    Returns
    -------
    ret: bool
        Whether the two types are equal or not.
    """
    if type(lhs) is not type(rhs):
        return False
    if isinstance(lhs, DataType) and isinstance(rhs, DataType):
        return lhs.name == rhs.name
    elif isinstance(lhs, PointerType) and isinstance(rhs, PointerType):
        return type_equal(lhs.base_type, rhs.base_type)
    elif isinstance(lhs, VoidType) and isinstance(rhs, VoidType):
        return True
    elif isinstance(lhs, TensorPointerType) and isinstance(rhs, TensorPointerType):
        return type_equal(lhs.tensor_type, rhs.tensor_type)
    elif isinstance(lhs, TensorType) and isinstance(rhs, TensorType):
        from hidet.ir.expr import is_constant

        if not type_equal(lhs.dtype, rhs.dtype):
            return False
        if len(lhs.shape) != len(rhs.shape):
            return False
        for a, b in zip(lhs.shape, rhs.shape):
            if is_constant(a) ^ is_constant(b):
                return False
            elif is_constant(a) and is_constant(b):
                if int(a) != int(b):
                    return False
            else:
                # we do not have equivalence checking for symbolic expression
                pass
        # do not check layout
        return True
    elif isinstance(lhs, FuncType) and isinstance(rhs, FuncType):
        assert lhs.param_types is not None and lhs.ret_type is not None
        assert rhs.param_types is not None and rhs.ret_type is not None
        if len(lhs.param_types) != len(rhs.param_types):
            return False
        if not type_equal(lhs.ret_type, rhs.ret_type):
            return False
        for a, b in zip(lhs.param_types, rhs.param_types):
            if not type_equal(a, b):
                return False
        return True
    else:
        raise NotImplementedError()


void_p = PointerType(VoidType())
byte_p = PointerType(data_type('uint8'))
void = VoidType()
