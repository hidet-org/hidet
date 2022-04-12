from typing import Sequence, Optional, Union, List, Tuple, Mapping, Callable

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
        assert name in ['host', 'global', 'shared', 'register']
        self.name = name


# scalar type and tensor type
class ScalarType(TypeNode):
    def __init__(self, name):
        if name:
            assert name in ['float32', 'int32', 'uint8', 'uint32', 'bool'], name
        self.name = name

    def nbytes(self) -> int:
        bytes_dict = {
            'float32': 4,
            'int32': 4,
            'uint8': 1,
            'uint32': 4,
            'bool': 1
        }
        return bytes_dict[self.name]


class TensorType(TypeNode):
    def __init__(self,
                 scope: Optional[Union[Scope, str]] = None,
                 dtype: Optional[Union[ScalarType, str]] = None,
                 shape: Optional[Sequence[Int]] = None,
                 layout: Optional[Union[Sequence[Int], 'DataLayout']] = None):
        from hidet.ir.expr import convert
        from hidet.ir.layout import DataLayout, StridesLayout
        if isinstance(scope, str):
            scope = Scope(scope)
        if isinstance(dtype, str):
            dtype = ScalarType(dtype)
        if shape:
            shape = [ir.convert(s) for s in shape]
        if isinstance(layout, (list, tuple)) and shape is not None:
            # use shape and strides to define layout
            strides = layout
            layout = StridesLayout(shape, strides)
        if layout is None and shape is not None and len(shape) > 0 and shape[0] is not None:
            layout = DataLayout.row_major(shape)
        if shape is None and isinstance(layout, DataLayout):
            # use shape in data layout
            shape = layout.shape

        self.scope: Scope = scope
        self.scalar_type: ScalarType = dtype
        self.shape: Tuple[Expr] = convert(shape)
        self.layout: DataLayout = layout

    def storage_bytes(self) -> Expr:
        return self.layout.size * self.scalar_type.nbytes()

    def slice_out(self, dims: Sequence[int]) -> 'TensorType':
        layout = self.layout.slice_out(dims)
        return TensorType(self.scope, self.scalar_type, layout=layout)

    def split(self, dim2factor: Mapping[int, Int]) -> 'TensorType':
        layout = self.layout.split(dim2factor)
        return TensorType(self.scope, self.scalar_type, layout=layout)

    def reorder(self, order: Sequence[int]):
        layout = self.layout.reorder(order)
        return TensorType(self.scope, self.scalar_type, layout=layout)


TypeLike = Union[str, TypeNode]


class FuncType(TypeNode):
    def __init__(self,
                 param_types: Optional[List[TypeLike]] = None,
                 ret_type: Optional[TypeLike] = None,
                 type_infer_func: Optional[Callable] = None):  # Callable[[a number of TypeNode], TypeNode]
        self.param_types = [self._convert_type(tp) for tp in param_types] if param_types else None
        self.ret_type = self._convert_type(ret_type) if ret_type else None
        self.type_infer_func = type_infer_func
        assert not all(v is None for v in [ret_type, type_infer_func]), 'Please provide either a static type or a type infer func'

    def ret_type_on(self, arg_types: List[TypeNode]) -> TypeNode:
        if self.ret_type is not None:
            # todo: add type checking
            return self.ret_type
        else:
            return self.type_infer_func(*arg_types)

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


def tensor_type(scope, dtype, shape=None, layout=None):
    return TensorType(scope, dtype, shape, layout)
