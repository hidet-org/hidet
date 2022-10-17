from __future__ import annotations
from typing import Sequence, Optional, Union, List, Tuple, Mapping, Callable, Iterable
import numpy as np

from hidet import ir
from hidet.ir.node import Node
from hidet.utils import initialize

# typing forward declaration
Expr = 'Expr'
Int = Union['Expr', int]


class TypeNode(Node):
    def __invert__(self) -> TypeNode:
        # get the pointer type that points to current type
        if isinstance(self, TensorType):
            return TensorPointerType.from_tensor_type(self)
        elif isinstance(self, ScalarType):
            return PointerType(base_type=self)
        elif isinstance(self, (PointerType, TensorPointerType)):
            return PointerType(base_type=self)
        else:
            raise ValueError('Can not recognize type {}'.format(self))




short2long = {
    'bf16': 'bfloat16',
    'tf32': 'tfloat32',
    'f16': 'float16',
    'fp16': 'float16',
    'f32': 'float32',
    'fp32': 'float32',
    'f64': 'float64',
    'fp64': 'float64'
}

dtype_list = [
    'float64',
    'float32',
    'tfloat32',
    'uint64',
    'int64',
    'uint32',
    'int32',
    'uint32',
    'uint16',
    'bfloat16',
    'float16',
    'uint8',
    'int8',
    'bool'
]

float_dtype_rank = {}


@initialize()
def init_float_dtype_rank():
    for idx, dtype in enumerate(dtype_list):
        float_dtype_rank[dtype] = len(dtype_list) - idx


class ScalarType(TypeNode):
    def __init__(self, name: str):
        if not isinstance(name, str):
            raise ValueError('Expect the name of scalar type.')
        if name in short2long:
            name = short2long[name]
        if name not in dtype_list:
            raise ValueError('Can not recognize data type {}, candidates:\n{}'.format(name, dtype_list))
        self.name = name

    def __eq__(self, other):
        if isinstance(other, str):
            other = ScalarType(other)
        return self.name == other.name

    def __ne__(self, other):
        if isinstance(other, str):
            other = ScalarType(other)
        return self.name != other.name

    def __le__(self, other):
        if isinstance(other, str):
            other = ScalarType(other)
        return float_dtype_rank[self.name] <= float_dtype_rank[other.name]

    def __lt__(self, other):
        if isinstance(other, str):
            other = ScalarType(other)
        return float_dtype_rank[self.name] < float_dtype_rank[other.name]

    def __ge__(self, other):
        if isinstance(other, str):
            other = ScalarType(other)
        return float_dtype_rank[self.name] >= float_dtype_rank[other.name]

    def __gt__(self, other):
        if isinstance(other, str):
            other = ScalarType(other)
        return float_dtype_rank[self.name] > float_dtype_rank[other.name]

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __getitem__(self, item) -> TensorType:
        if not isinstance(item, tuple):
            item = (item,)
        return tensor_type(dtype=self, shape=list(item), layout=None)

    @staticmethod
    def from_numpy_dtype(np_dtype):
        if np_dtype == np.float32:
            return ScalarType('float32')
        elif np_dtype == np.int32:
            return ScalarType('int32')
        elif np_dtype == np.int64:
            return ScalarType('int64')
        else:
            raise ValueError("Unrecognized numpy data type: '{}'".format(np_dtype))

    @staticmethod
    def float16() -> ScalarType:
        return ScalarType('float16')

    @staticmethod
    def float32() -> ScalarType:
        return ScalarType('float32')

    @staticmethod
    def int32() -> ScalarType:
        return ScalarType('int32')

    @staticmethod
    def int64() -> ScalarType:
        return ScalarType('int64')

    @staticmethod
    def uint32() -> ScalarType:
        return ScalarType('uint32')

    @staticmethod
    def uint64() -> ScalarType:
        return ScalarType('uint64')

    @staticmethod
    def uint8() -> ScalarType:
        return ScalarType('uint8')

    def nbytes(self) -> int:
        bytes_dict = {
            'uint64': 8,
            'uint32': 4,
            'uint16': 2,
            'uint8': 1,
            'float64': 8,
            'float32': 4,
            'float16': 2,
            'float8': 1,
            'int64': 8,
            'int32': 4,
            'int16': 2,
            'int8': 1,
            'bool': 1,
            'tfloat32': 4,
            'bfloat16': 2,
        }
        return bytes_dict[self.name]

    def is_float(self) -> bool:
        return self.name in ['float16', 'bfloat16', 'float32', 'float64']

    def is_integer(self) -> bool:
        return self.name in ['bool', 'uint8', 'int32', 'uint32', 'int64']

    def min_value(self) -> Expr:
        from hidet.ir.expr import Constant
        value_dict = {
            'float16': -65504,
            'float32': -3.4e38,
            'float64': -1e308,
            'int64': -9223372036854775808 + 1,
            'int32': -2147483648 + 1,
            'uint32': 0
        }
        if self.name not in value_dict:
            raise NotImplementedError(self.name)
        return Constant(value_dict[self.name], self)

    def max_value(self) -> Expr:
        from hidet.ir.expr import Constant
        value_dict = {
            'float16': 65504,
            'float32': 3.4e38,
            'float64': 1e308,
            'int64': 9223372036854775807,
            'int32': 2147483647,
            'uint32': 4294967295
        }
        if self.name not in value_dict:
            raise NotImplementedError()
        return Constant(value_dict[self.name], self)

    def zero(self) -> Expr:
        from hidet.ir.expr import Constant
        return Constant(0, self)

    def one(self) -> Expr:
        from hidet.ir.expr import Constant
        return Constant(1, self)

    @staticmethod
    def resolve_out_dtype(lhs: Union[ScalarType, str], rhs: Union[ScalarType, str]) -> ScalarType:
        lhs = ScalarType(lhs) if isinstance(lhs, str) else lhs
        rhs = ScalarType(rhs) if isinstance(rhs, str) else rhs
        if lhs.is_float() and rhs.is_float():
            nbytes = max(lhs.nbytes(), rhs.nbytes())
            return ScalarType('float{}'.format(nbytes * 8))
        elif lhs.is_integer() and rhs.is_integer():
            nbytes = max(lhs.nbytes(), rhs.nbytes())
            return ScalarType('int{}'.format(nbytes * 8))
        else:
            raise NotImplementedError('resolve out dtype for {} and {}'.format(lhs, rhs))


class TensorType(TypeNode):
    def __init__(self,
                 # scope: Optional[Scope] = None,
                 dtype: Optional[ScalarType] = None,
                 shape: Optional[Tuple[Expr, ...]] = None,
                 layout: Optional['DataLayout'] = None):
        from hidet.ir.layout import DataLayout
        # self.scope: Scope = scope
        self.scalar_type: ScalarType = dtype
        self.shape: Tuple[Expr] = shape
        self.layout: DataLayout = layout

    def storage_bytes(self) -> Expr:
        return self.layout.size * self.scalar_type.nbytes()

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


class VoidType(TypeNode):
    pass


class PointerType(TypeNode):
    def __init__(self, base_type, specifiers: Optional[Sequence[str]] = None, use_bracket: bool = False):
        super().__init__()
        if isinstance(base_type, str):
            base_type = ScalarType(base_type)
        self.base_type: TypeNode = base_type
        self.specifiers: List[str] = list(specifiers) if specifiers else []
        self.use_bracket: bool = use_bracket


class ReferenceType(TypeNode):
    def __init__(self, base_type):
        super().__init__()
        self.base_type = base_type


class TensorPointerType(TypeNode):
    def __init__(self,
                 dtype: Optional[Union[ScalarType, str]] = None,
                 shape: Optional[Sequence[Int]] = None,
                 layout: Optional[Union[Sequence[Int], 'DataLayout']] = None):
        self.tensor_type: TensorType = tensor_type(dtype, shape, layout)

    @staticmethod
    def from_tensor_type(tp: TensorType) -> TensorPointerType:
        tpt = object.__new__(TensorPointerType)
        tpt.tensor_type = tp
        return tpt


TypeLike = Union[str, TypeNode]


class FuncType(TypeNode):
    def __init__(self,
                 param_types: Optional[List[TypeLike]] = None,
                 ret_type: Optional[TypeLike] = None,
                 type_infer_func: Optional[Callable] = None):  # Callable[[a number of TypeNode], TypeNode]
        self.param_types = [self._convert_type(tp) for tp in param_types] if param_types is not None else None
        self.ret_type = self._convert_type(ret_type) if ret_type is not None else None
        self.type_infer_func = type_infer_func
        assert not all(v is None for v in [ret_type, type_infer_func]), 'Please provide either a static type or a type infer func'

    def ret_type_on(self, arg_types: List[TypeNode]) -> TypeNode:
        if self.ret_type is not None:
            # todo: add type checking
            return self.ret_type
        else:
            return self.type_infer_func(arg_types)

    def _convert_type(self, tp: Union[str, TypeNode]):
        if isinstance(tp, str):
            return ScalarType(tp)
        else:
            return tp

    @staticmethod
    def from_func(func):
        return FuncType([param.type for param in func.params], func.ret_type)


def scalar_type(type_name):
    return ScalarType(type_name)


def tensor_type(dtype, shape: Optional[Sequence[Union[int, Expr]]] = None, layout: Optional['DataLayout'] = None):
    """
    Construct a tensor type. Shape and layout must be given at least one.

    Parameters
    ----------
    dtype: str or ScalarType
        The scalar type of this tensor.

    shape: Optional[List[Union[int, Expr]]]
        The shape of the tensor. If not given, the shape in layout will be used.

    layout: Optional[DataLayout]
        The layout of the tensor. If not given, the row major layout of given shape will
        be used.

    Returns
    -------
    ret: TensorType
        The constructed tensor type
    """
    from hidet.ir.expr import convert, Constant
    from hidet.ir.layout import DataLayout, StridesLayout
    if isinstance(dtype, str):
        dtype = ScalarType(dtype)
    if not isinstance(dtype, ScalarType):
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
                raise ValueError('The shape of tensor and the shape of layout are not compatible, {} vs {}'.format(list(shape), list(layout.shape)))
    shape = convert(shape)
    return TensorType(dtype, shape, layout)


def pointer_type(base_type):
    return PointerType(base_type)


def void_pointer():
    return PointerType(VoidType())


void_p = PointerType(VoidType())

int32 = ScalarType('int32')
uint32 = ScalarType('uint32')
int64 = ScalarType('int64')
uint64 = ScalarType('uint64')
uint8 = ScalarType('uint8')
boolean = ScalarType('bool')
