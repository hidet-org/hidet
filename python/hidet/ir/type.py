from __future__ import annotations
from typing import Sequence, Optional, Union, List, Tuple, Mapping, Callable, Iterable
import numpy as np

from hidet import ir
from hidet.ir.node import Node

# typing forward declaration
Expr = 'Expr'
Int = Union['Expr', int]


class TypeNode(Node):
    pass


# scope
class Scope(Node):
    def __init__(self, name):
        assert name in ['host', 'global', 'shared', 'register', 'unspecified']
        self.name = name


dtype_list = [
    'int64',
    'float64',
    'int32',
    'uint32',
    'float32',
    'tfloat32',
    'bfloat16',
    'int32',
    'float16',
    'uint8',
    'bool'
]

float_dtype_rank = {}
for idx, dtype in enumerate(dtype_list):
    float_dtype_rank[dtype] = len(dtype_list) - idx


class ScalarType(TypeNode):
    def __init__(self, name):
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

    def __hash__(self):
        return hash(self.name)

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
            'float32': 4,
            'tfloat32': 4,
            'bfloat16': 2,
            'float16': 2,
            'int32': 4,
            'uint8': 1,
            'uint32': 4,
            'int64': 8,
            'bool': 1
        }
        return bytes_dict[self.name]

    def is_float(self) -> bool:
        return self.name in ['float16', 'bfloat16', 'float32', 'float64']

    def is_integer(self) -> bool:
        return self.name in ['bool', 'uint8', 'int32', 'uint32', 'int64']

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
                 scope: Optional[Scope] = None,
                 dtype: Optional[ScalarType] = None,
                 shape: Optional[Tuple[Expr, ...]] = None,
                 layout: Optional['DataLayout'] = None):
        from hidet.ir.layout import DataLayout
        self.scope: Scope = scope
        self.scalar_type: ScalarType = dtype
        self.shape: Tuple[Expr] = shape
        self.layout: DataLayout = layout

    def storage_bytes(self) -> Expr:
        return self.layout.size * self.scalar_type.nbytes()

    def slice_out(self, dims: Sequence[int]) -> 'TensorType':
        layout = self.layout.slice_out(dims)
        return tensor_type(self.scope, self.scalar_type, layout=layout)

    def split(self, dim2factor: Mapping[int, Int]) -> 'TensorType':
        layout = self.layout.split(dim2factor)
        return tensor_type(self.scope, self.scalar_type, layout=layout)

    def reorder(self, order: Sequence[int]):
        layout = self.layout.reorder(order)
        return tensor_type(self.scope, self.scalar_type, layout=layout)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


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


def tensor_type(scope, dtype, shape: Optional[List[Union[int, Expr]]] = None, layout: Optional['DataLayout'] = None):
    """
    Construct a tensor type. Shape and layout must be given at least one.

    Parameters
    ----------
    scope: str or Scope
        The scope of the tensor. Scope can be 'host', 'global', 'shared', and 'local'

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
    if isinstance(scope, str):
        scope = Scope(scope)
    if not isinstance(scope, Scope):
        raise ValueError('Tensor type scope expect a "str" or "Scope", but got {}'.format(type(scope)))
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
        layout = DataLayout.row_major([int(v) for v in shape])
    else:
        assert isinstance(layout, DataLayout)
        assert isinstance(shape, (list, tuple))
        for a, b in zip(shape, layout.shape):
            assert int(a) == int(b)
    shape = convert(shape)
    return TensorType(scope, dtype, shape, layout)


def max_float_dtype(float_dtypes: Iterable[str]) -> str:
    return max(float_dtypes, key=lambda dtype: float_dtype_rank[dtype])
