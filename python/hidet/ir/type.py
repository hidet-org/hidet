import functools
import operator
from typing import Sequence, Optional, Union, Dict, Tuple, List, Callable
from hidet.ir.node import Node


class TypeNode(Node):
    pass


# scope
class Scope:
    def __init__(self, name):
        assert name in ['host', 'global', 'shared', 'register']
        self.name = name


# data layout
Int = Union['Expr', int]


class DataLayout:
    def __init__(self, shape=None, global2local=None):
        self.shape: List[Int] = shape
        self.global2local: Callable[[Int, ...], Int] = global2local

    def serialize(self, *args: Sequence[Int]) -> Int:
        if self.global2local is None:
            raise NotImplementedError()
        else:
            return self.global2local(*args)


class StridesLayout(DataLayout):
    def __init__(self, shape, strides):
        from hidet.ir.expr import convert
        super().__init__(shape=shape,
                         global2local=lambda *indices: sum(v * self.strides[i] for i, v in enumerate(indices)))
        self.strides: List[Int] = [convert(v) for v in strides]

    def __add__(self, other):
        assert isinstance(other, StridesLayout)
        return StridesLayout.combine(self, other)

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
        nshape = [shape[pr[0]] for pr in pairs]
        strides = [None] * rank
        for i in range(rank):
            strides[i] = functools.reduce(operator.mul, nshape[i + 1:], 1)
        return StridesLayout(strides)

    @staticmethod
    def combine(lhs: 'StridesLayout', rhs: 'StridesLayout') -> 'StridesLayout':
        shape = lhs.shape + rhs.shape
        rhs_size = functools.reduce(operator.mul, rhs.shape)
        strides = [v * rhs_size for v in lhs.strides] + rhs.strides
        return StridesLayout(shape, strides)


class LocalLayout(DataLayout):
    def __init__(self, local_size, shape, global2local):
        super().__init__(shape, global2local)
        self.local_size: int = local_size

    def serialize(self, *args: Sequence[Int]) -> Int:
        return self.global2local(*args)

    def __add__(self, other):
        return LocalLayout.combine(self, other)

    def __radd__(self, other):
        return LocalLayout.combine(other, self)

    @staticmethod
    def combine(lhs: DataLayout, rhs: DataLayout) -> 'LocalLayout':
        lhs_rank = len(lhs.shape)
        rhs_rank = len(rhs.shape)
        shape = lhs.shape + rhs.shape
        if isinstance(lhs, StridesLayout) and isinstance(rhs, StridesLayout):
            return StridesLayout.combine(lhs, rhs)
        elif isinstance(lhs, StridesLayout) and isinstance(rhs, LocalLayout):
            rhs_size = rhs.local_size
            local_size = functools.reduce(operator.mul, lhs.shape) * rhs.local_size
        elif isinstance(lhs, LocalLayout) and isinstance(rhs, StridesLayout):
            rhs_size = functools.reduce(operator.mul, rhs.shape)
            local_size = lhs.local_size * functools.reduce(operator.mul, rhs.shape)
        else:
            raise ValueError('Can not combine two local layouts.')

        def global2local(*args):
            return lhs.global2local(*args[:lhs_rank]) * rhs_size + rhs.global2local(*args[lhs_rank:])

        return LocalLayout(local_size, shape, global2local)


# scalar type and tensor type
class ScalarType(TypeNode):
    def __init__(self, name):
        if name:
            assert name in ['float32', 'int32', 'bool'], name
        self.name = name


class TensorType(TypeNode):
    def __init__(self,
                 scope: Optional[Union[Scope, str]] = None,
                 dtype: Optional[Union[ScalarType, str]] = None,
                 shape: Optional[Sequence[Int]] = None,
                 layout: Optional[Union[Sequence[Int], DataLayout]] = None):
        from hidet.ir.expr import convert
        if isinstance(scope, str):
            scope = Scope(scope)
        if isinstance(dtype, str):
            dtype = ScalarType(dtype)
        if layout:
            if isinstance(layout, (list, tuple)):
                strides = layout
                layout = StridesLayout(shape, strides)
        if shape:
            shape = [convert(s) for s in shape]
        self.scope: Scope = scope
        self.scalar_type: ScalarType = dtype
        self.shape = shape
        self.layout = layout

    def nbytes(self):
        from hidet.ir.expr import convert, Constant
        from hidet.ir.functors import simplify
        max_index_expr = sum([(a - 1) * b for a, b in zip(self.shape, self.layout)], convert(0))
        max_index_value = simplify(max_index_expr)
        if isinstance(max_index_value, Constant):
            return max_index_value.value
        else:
            raise Exception("Can only calculate size of static tensor.")


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
