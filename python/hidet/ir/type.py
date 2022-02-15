from typing import Sequence, Optional, Union, List

from hidet import ir
from hidet.ir.node import Node

# typing forward declaration
Expr = 'Expr'
Int = Union['Expr', int]


class TypeNode(Node):
    pass


# scope
class Scope:
    def __init__(self, name):
        assert name in ['host', 'global', 'shared', 'register']
        self.name = name


# scalar type and tensor type
class ScalarType(TypeNode):
    def __init__(self, name):
        if name:
            assert name in ['float32', 'int32', 'uint8', 'bool'], name
        self.name = name

    def nbytes(self) -> int:
        bytes_dict = {
            'float32': 4,
            'int32': 4,
            'uint8': 1,
            'bool': 1
        }
        return bytes_dict[self.name]


class TensorType(TypeNode):
    def __init__(self,
                 scope: Optional[Union[Scope, str]] = None,
                 dtype: Optional[Union[ScalarType, str]] = None,
                 shape: Optional[Sequence[Int]] = None,
                 layout: Optional[Union[Sequence[Int], 'DataLayout']] = None):
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
        if shape is None and isinstance(layout, DataLayout):
            # use shape in data layout
            shape = layout.shape

        self.scope: Scope = scope
        self.scalar_type: ScalarType = dtype
        self.shape: List[Expr] = shape
        self.layout: DataLayout = layout

    def storage_bytes(self) -> Expr:
        return self.layout.size * self.scalar_type.nbytes()

    def slice_out(self, dims: Sequence[int]) -> 'TensorType':
        layout = self.layout.slice_out(dims)
        return TensorType(self.scope, self.scalar_type, layout=layout)


class FuncType(TypeNode):
    def __init__(self, param_types, ret_type):
        self.param_types = param_types
        self.ret_type = ret_type

    @staticmethod
    def from_func(func):
        return FuncType([param.type for param in func.params], func.ret_type)


def scalar_type(type_name):
    return ScalarType(type_name)


def tensor_type(scope, dtype, shape, layout):
    return TensorType(scope, dtype, shape, layout)
