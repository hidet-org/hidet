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
    def __init__(self, name: str):
        if self.__class__ is DataType:
            raise TypeError('DataType is an abstract class. Please use data_type() to create a concrete DataType.')

        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, DataType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __call__(self, value: Any):
        return self.constant(value)

    def short_name(self) -> str:
        raise NotImplementedError()

    def nbytes(self) -> int:
        raise NotImplementedError()

    def is_float(self) -> bool:
        raise NotImplementedError()

    def is_integer(self) -> bool:
        raise NotImplementedError()

    def is_vector(self) -> bool:
        raise NotImplementedError()

    def constant(self, value: Any):
        raise NotImplementedError()

    def one(self):
        raise NotImplementedError()

    def zero(self):
        raise NotImplementedError()

    def min_value(self):
        raise NotImplementedError()

    def max_value(self):
        raise NotImplementedError()


class TensorType(TypeNode):
    def __init__(
        self,
        dtype: Optional[DataType] = None,
        shape: Optional[Tuple[Expr, ...]] = None,
        layout: Optional['DataLayout'] = None,
    ):
        from hidet.ir.layout import DataLayout

        self.dtype: DataType = dtype
        self.shape: Tuple[Expr] = shape
        self.layout: DataLayout = layout

    def storage_bytes(self) -> Expr:
        return self.layout.size * self.dtype.nbytes()

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
        self.specifiers: List[str] = list(specifiers) if specifiers else []
        self.use_bracket: bool = use_bracket


class ReferenceType(TypeNode):
    def __init__(self, base_type):
        super().__init__()
        self.base_type = base_type


class TensorPointerType(TypeNode):
    def __init__(
        self,
        dtype: Optional[Union[DataType, str]] = None,
        shape: Optional[Sequence[Int]] = None,
        layout: Optional[Union[Sequence[Int], 'DataLayout']] = None,
    ):
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
        type_infer_func: Optional[Callable] = None,
    ):  # Callable[[a number of TypeNode], TypeNode]
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


def tensor_type(dtype, shape=None, layout=None):
    """
    Construct a tensor type.

    Shape and layout must be given at least one.

    Parameters
    ----------
    dtype: str or DataType
        The scalar type of this tensor.

    shape: Optional[Sequence[Union[int, Expr]]]
        The shape of the tensor. If not given, the shape in layout will be used.

    layout: Optional[hidet.ir.layout.DataLayout]
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


def void_pointer():
    return PointerType(VoidType())


def data_type(name: str) -> DataType:
    from hidet.ir.dtypes import name2dtype, sname2dtype

    if name in name2dtype:
        return name2dtype[name]
    elif name in sname2dtype:
        return sname2dtype[name]
    else:
        raise ValueError('Unknown data type: {}'.format(name))


void_p = PointerType(VoidType())
