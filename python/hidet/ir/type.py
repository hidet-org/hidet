import functools
import operator
from typing import Sequence, Optional, Union, Dict, Tuple, List, Callable
from hidet.ir.node import Node
from hidet import ir


class TypeNode(Node):
    pass


# scope
class Scope:
    def __init__(self, name):
        assert name in ['host', 'global', 'shared', 'register']
        self.name = name


# typing forward declaration
Expr = 'Expr'
Int = Union['Expr', int]


# data layout
class DataLayout:
    def __init__(self, shape=None, size=None, global2local=None):
        self.shape: List[Int] = shape
        self.size: Int = size
        self.global2local: Callable[[Int, ...], Int] = global2local

    def __call__(self, *args: Sequence[Int]):
        return self.serialize(*args)

    def __add__(self, other):
        return DataLayout.concat(lhs=self, rhs=other)

    def __mul__(self, other):
        return DataLayout.product(outer=self, inner=other)

    def serialize(self, *args: Sequence[Int]):
        assert len(args) == len(self.shape)
        scalar_index = self.global2local(*args)
        if isinstance(scalar_index, int) and isinstance(self.size, int):
            assert scalar_index < self.size
        return scalar_index

    @staticmethod
    def product(outer: 'DataLayout', inner: 'DataLayout'):
        assert len(outer.shape) == len(inner.shape)
        shape = [a * b for a, b in zip(outer.shape, inner.shape)]
        size = outer.size * inner.size

        def global2local(*args):
            lhs_args = [v // b for v, b in zip(args, inner.shape)]
            rhs_args = [v % b for v, b in zip(args, inner.shape)]
            return outer.global2local(*lhs_args) * inner.size + inner.global2local(*rhs_args)

        return DataLayout(shape, size, global2local)

    @staticmethod
    def concat(lhs: 'DataLayout', rhs: 'DataLayout'):
        shape = list(lhs.shape) + list(rhs.shape)
        size = lhs.size * rhs.size

        def global2local(*args):
            lhs_args = args[:len(lhs.shape)]
            rhs_args = args[len(lhs.shape):]
            return lhs.global2local(*lhs_args) * rhs.size + rhs.global2local(*rhs_args)

        return DataLayout(shape, size, global2local)

    @staticmethod
    def local_layout(shape: Sequence[Int]):
        return DataLayout(shape, size=1, global2local=lambda *args: 0)


class StridesLayout(DataLayout):
    def __init__(self, shape, strides):
        super().__init__(shape=shape,
                         size=StridesLayout.storage_size(shape, strides),
                         global2local=lambda *indices: sum(v * self.strides[i] for i, v in enumerate(indices)))
        self.strides: List[Int] = strides

    @staticmethod
    def storage_size(shape, strides) -> Expr:
        # assume the strides are positive, but do not assume the tensor is contiguous.
        max_index = sum([(a - ir.convert(1)) * b for a, b in zip(shape, strides)]) + 1
        return ir.functors.simplify(max_index)

    @staticmethod
    def row_major(shape: Sequence[Int]) -> 'StridesLayout':
        return StridesLayout.from_shape(shape, list(range(len(shape))))

    @staticmethod
    def column_major(shape: Sequence[Int]) -> 'StridesLayout':
        return StridesLayout.from_shape(shape, list(reversed(range(len(shape)))))

    @staticmethod
    def from_shape(shape: Sequence[Int], perm: Sequence[int]):
        assert len(shape) == len(perm)
        rank = len(shape)
        pairs = [(i, p) for i, p in zip(range(rank), perm)]
        pairs = sorted(pairs, key=lambda pr: pr[1])
        new_shape = [shape[pr[0]] for pr in pairs]
        strides = [None] * rank
        for i in range(rank):
            strides[i] = functools.reduce(operator.mul, new_shape[i + 1:], 1)
        return StridesLayout(shape, strides)


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
                 layout: Optional[Union[Sequence[Int], DataLayout]] = None):
        if isinstance(scope, str):
            scope = Scope(scope)
        if isinstance(dtype, str):
            dtype = ScalarType(dtype)
        if layout:
            if isinstance(layout, (list, tuple)):
                strides = layout
                layout = StridesLayout(shape, strides)
        if shape:
            shape = [ir.convert(s) for s in shape]
        self.scope: Scope = scope
        self.scalar_type: ScalarType = dtype
        self.shape: List[Expr] = shape
        self.layout: DataLayout = layout

    def storage_bytes(self) -> Expr:
        return self.layout.size * self.scalar_type.nbytes()


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
