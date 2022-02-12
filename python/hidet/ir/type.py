from typing import Sequence, Optional, Union, Dict, Tuple, List, Callable, Mapping
import functools
import operator
from hidet.ir.node import Node
from hidet import ir
from hidet.utils import prod


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

    def tile(self, inner_shape: Sequence[Int]) -> 'DataLayout':
        assert len(inner_shape) == len(self.shape)
        assert all(b % a == 0 for a, b in zip(inner_shape, self.shape) if isinstance(a, int) and isinstance(b, int))
        shape = [b // a for a, b in zip(inner_shape, self.shape)] + inner_shape

        def global2local(*args):
            rank = len(inner_shape)
            assert len(args) == rank * 2
            outer_args = args[:rank]
            inner_args = args[rank:]
            merged_args = [o * inner_shape[r] + i for r, o, i in zip(range(rank), outer_args, inner_args)]
            return self.global2local(*merged_args)

        return DataLayout(shape, self.size, global2local)

    def split(self, dim2factor: Mapping[int, Int]) -> 'DataLayout':
        """
        3-dimension tensor with shape [a, b, c]
        after split(dim2factor={0: 2, 1: 3}) got
        5-dimension tensor with shape [(a + 1) // 2, 2, (b + 2) // 3, 3, c]
        """
        shape = []
        for i, s in enumerate(self.shape):
            if i in dim2factor:
                factor = dim2factor[i]
                outer = (s + factor - 1) // factor
                shape.extend([outer, factor])
            else:
                shape.append(s)

        def global2local(*args):
            assert len(args) == len(shape)
            merged_args = []
            c = 0
            for i in range(len(self.shape)):
                if i in dim2factor:
                    outer_idx = args[c]
                    inner_idx = args[c+1]
                    merged_args.append(outer_idx * dim2factor[i] + inner_idx)
                    c += 2
                else:
                    merged_args.append(args[c])
                    c += 1
            return self.global2local(*merged_args)

        return DataLayout(shape, self.size, global2local)

    def reorder(self, order: Sequence[int]):
        """
        3-dimension tensor with shape [a, b, c]
        after reorder([0, 2, 1]) got
        3-dimension tensor with shape [a, c, b]

        It is a special case of fuse.
        """
        return self.fuse(order)

    def fuse(self, dim2fuse: Sequence[Union[Sequence[int], int]]):
        """
        3-dimension tensor with shape [a, b, c]
        after fuse([2, [1, 0]]) got
        3-dimension tensor with shape [c, b * a]
        (i, j, k) of the result data layout will be mapped to (k, j * I + i) of the original data layout
        """
        covered = []
        shape = []
        dims = []
        for i in range(len(dim2fuse)):
            item = dim2fuse[i]
            if isinstance(item, int):
                item = [item]
            else:
                item = list(item)
            dims.append(item)
            covered.extend(item)
            shape.append(prod(item))
        assert len(covered) == len(self.shape) and len(set(covered)) == len(covered), "missing some dimension or duplicated dimension"

        def global2local(*args):
            original_args = [None] * len(self.shape)
            for i in range(len(shape)):
                dim_sizes = [self.shape[v] for v in dims[i]]
                for j, dim in enumerate(dims[i]):
                    original_args[dim] = args[i] // prod(dim_sizes[j + 1:]) % dim_sizes[j]
            return self.global2local(*original_args)

        return DataLayout(shape, self.size, global2local)

    def slice_out(self, dims: Sequence[int]):
        """
        3-dimension tensor with shape [a, b, c]
        after cut({0, 2}) got
        1-dimension tensor with shape [b]
        """
        dims = set(dims)
        assert all(d < len(self.shape) for d in dims)
        shape = [s for r, s in enumerate(self.shape) if r not in dims]

        def global2local(*args):
            assert len(args) == len(shape)
            merged_args = []
            c = 0
            for i in range(len(self.shape)):
                if i in dims:
                    merged_args.append(0)
                else:
                    merged_args.append(args[c])
                    c += 1
            return self.global2local(*merged_args)
        return DataLayout(shape, self.size, global2local)

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
    def local(shape: Sequence[Int]):
        return DataLayout(shape, size=1, global2local=lambda *args: 0)

    @staticmethod
    def row_major(shape: Sequence[Int]):
        return StridesLayout.row_major(shape)

    @staticmethod
    def column_major(shape: Sequence[Int]) -> 'StridesLayout':
        return StridesLayout.column_major(shape)


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
        tuples = [[i, p, None] for i, p in zip(range(rank), perm)]
        tuples = sorted(tuples, key=lambda t: t[1])
        reordered_shape = [shape[t[0]] for t in tuples]
        for i in range(rank):
            tuples[i][2] = prod(reordered_shape[i+1:])
        tuples = sorted(tuples, key=lambda t: t[0])
        strides = [t[2] for t in tuples]
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
