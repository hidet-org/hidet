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
import itertools
from collections import OrderedDict
from typing import Sequence, Union, List, Mapping, Dict, Tuple, Optional

from hidet.ir.node import Node
from hidet.utils import prod

# typing forward declaration
Expr = 'Expr'
Int = Union['Expr', int]
Bool = Union['Expr', bool]


def is_power_of_two(n: int):
    return n != 0 and (n & (n - 1)) == 0


def is_atom(expr: Expr):
    from hidet.ir import Constant, Var

    return isinstance(expr, (Constant, Var))


def variablize(expr_list: Sequence[Expr], var2value: Dict['Var', Expr]) -> List['Var']:
    from hidet.ir import var

    out = []
    for expr in expr_list:
        if is_atom(expr):
            out.append(expr)
        else:
            v = var('v')
            var2value[v] = expr
            out.append(v)
    return out


def concat_let_expr(var2value, body: Expr):
    from hidet.ir import Let

    for var, value in reversed(var2value.items()):
        body = Let(var, value, body)
    return body


def to_data_layout(obj):
    if isinstance(obj, (tuple, list)):
        assert all(isinstance(v, int) for v in obj)
        return DataLayout.row_major(obj)
    elif isinstance(obj, DataLayout):
        return obj
    else:
        raise ValueError('Can not convert {} to a DataLayout, expect a list or tuple of ints'.format(obj))


# data layout
class DataLayout(Node):
    def __init__(self, shape=None, size=None):
        from hidet import ir

        if shape is None:
            shape = []
        self.shape: Tuple[Int] = tuple(int(v) if isinstance(v, ir.Constant) else v for v in shape)
        self.size: Int = size

    def __call__(self, *args: Int):
        return self.serialize(*args)

    def __add__(self, other):
        return DataLayout.concat(lhs=self, rhs=other)

    def __radd__(self, other):
        return DataLayout.concat(lhs=other, rhs=self)

    def __mul__(self, other):
        return DataLayout.product(outer=self, inner=other)

    def __str__(self):
        import numpy as np

        if int(self.size) > 1024:
            return '{}(shape={}, size={})'.format(self.__class__.__name__, self.shape, self.size)
        else:
            shape = [int(v) for v in self.shape]
            table = np.zeros(shape=shape, dtype=np.int)
            ranges = [range(v) for v in shape]
            for indices in itertools.product(*ranges):
                local_index = self.global2local(*indices)
                table[indices] = int(local_index)
            return np.array_str(table, max_line_width=120)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]

    def global2local(self, *args: Int) -> Int:
        raise NotImplementedError()

    def global2cond(self, *args: Int) -> Bool:
        raise NotImplementedError()

    def serialize(self, *args: Int):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            # support usage such as within_bound([1, 2, 3])
            args = args[0]
        assert len(args) == len(self.shape)
        # var2value = OrderedDict()
        # arg_vars = variablize(args, var2value)
        # scalar_index = self.global2local(*arg_vars)
        # scalar_index = concat_let_expr(var2value=var2value, body=scalar_index)
        scalar_index = self.global2local(*args)
        return scalar_index

    def within_bound(self, *args: Int):
        if isinstance(args[0], (tuple, list)) and len(args) == 1:
            # support usage such as within_bound([1, 2, 3])
            args = args[0]
        assert len(args) == len(self.shape)
        var2value = OrderedDict()
        arg_vars = variablize(args, var2value)
        cond = self.global2cond(*arg_vars)
        cond = concat_let_expr(var2value=var2value, body=cond)
        return cond

    def tile(self, inner_shape: Sequence[Int]):
        return TiledDataLayout(base=self, inner_shape=inner_shape)

    def split(self, dim2factor: Mapping[int, Int]):
        return SplitDataLayout(base=self, dim2factor=dim2factor)

    def reorder(self, order: Sequence[int]):
        return self.fuse(order)

    def swizzle(self, dim: int, regards_dim: Optional[int] = None, log_step: int = 0):
        return SwizzleDataLayout(base=self, dim=dim, regards_dim=regards_dim, log_step=log_step)

    def fuse(self, dim2fuse: Sequence[Union[Sequence[int], int]]):
        return FusedDataLayout(base=self, dim2fuse=dim2fuse)

    def slice_out(self, dims: Sequence[int]):
        return SliceOutDataLayout(base=self, dims=dims)

    @staticmethod
    def product(outer, inner):
        return ProductDataLayout(outer, inner)

    @staticmethod
    def concat(lhs, rhs):
        lhs = to_data_layout(lhs)
        rhs = to_data_layout(rhs)
        return ConcatDataLayout(lhs, rhs)

    @staticmethod
    def local(shape: Sequence[Int]):
        return LocalLayout(shape=shape)

    @staticmethod
    def row_major(shape: Sequence[Int]):
        return RowMajorLayout(shape)

    @staticmethod
    def column_major(shape: Sequence[Int]):
        return ColumnMajorLayout(shape)


class StridesLayout(DataLayout):
    def __init__(self, shape, strides):
        super().__init__(shape=shape, size=StridesLayout.storage_size(shape, strides))
        self.strides: List[Int] = strides

    def global2local(self, *args: Int) -> Int:
        return sum(v * self.strides[i] for i, v in enumerate(args))

    def global2cond(self, *args: Int) -> Bool:
        from hidet.ir.expr import LogicalAnd

        return LogicalAnd.join_list([v < s for s, v in zip(self.shape, args)])

    @staticmethod
    def storage_size(shape, strides) -> Expr:
        # assume the strides are positive, but do not assume the tensor is contiguous.
        from hidet.ir.functors import simplify

        max_index = sum((a - 1) * b for a, b in zip(shape, strides)) + 1
        return simplify(max_index)

    @staticmethod
    def from_shape(shape: Sequence[Int], perm: Sequence[int]):
        return StridesLayout(shape, StridesLayout.shape2strides(shape, perm))

    @staticmethod
    def shape2strides(shape: Sequence[Int], perm: Sequence[int]):
        assert len(shape) == len(perm)
        rank = len(shape)
        tuples = [[i, p, None] for i, p in zip(range(rank), perm)]
        tuples = sorted(tuples, key=lambda t: t[1])
        reordered_shape = [shape[t[0]] for t in tuples]
        for i in range(rank):
            tuples[i][2] = prod(reordered_shape[i + 1 :])
        tuples = sorted(tuples, key=lambda t: t[0])
        strides = [t[2] for t in tuples]
        return strides


class RowMajorLayout(StridesLayout):
    def __init__(self, shape):
        super().__init__(shape, StridesLayout.shape2strides(shape, list(range(len(shape)))))


class ColumnMajorLayout(StridesLayout):
    def __init__(self, shape):
        super().__init__(shape, StridesLayout.shape2strides(shape, list(reversed(range(len(shape))))))


class LocalLayout(DataLayout):
    def __init__(self, shape):
        super().__init__(shape=shape, size=1)

    def global2local(self, *args: Int) -> Int:
        return 0

    def global2cond(self, *args: Int) -> Bool:
        from hidet.ir.expr import LogicalAnd

        return LogicalAnd.join_list([v < s for s, v in zip(self.shape, args)])


class SwizzleDataLayout(DataLayout):
    """
    Swizzle a layout (called base layout) to get a swizzled data layout. The shape of swizzled layout is the same as
    the base layout.

    Example:
        A 2-dimension tensor with shape [a, b] where a = 2^m for some m and b <= a,
        After swizzle(plan={0: [1]}), we get a data layout with shape [a, b], and
          swizzled_layout(i, j) = base_layout(i ^ j, j)
        (Note, swizzle requires the swizzled dimension to be a power of 2)
    """

    def __init__(self, base: DataLayout, dim: int, regards_dim: Optional[int] = None, log_step: int = 0):
        self.base: DataLayout = base
        self.dim: int = int(dim)
        if regards_dim is None:
            if len(base.shape) != 2:
                raise ValueError(
                    'Optional regards_dim is only available for 2-rank layout, '
                    'got layout with shape {}.'.format(base.shape)
                )
            self.regards_dim = 1 - dim
        else:
            self.regards_dim = regards_dim
        self.log_step = log_step

        if self.dim == self.regards_dim:
            raise ValueError(
                'The swizzle dim and regards dim can not be the same, got {} and {}'.format(self.dim, self.regards_dim)
            )
        rank = len(base.shape)
        if not (0 <= self.dim < rank and 0 <= self.regards_dim < rank):
            raise ValueError(
                'The dim {} (regards dim {}) out of bound for layout {}'.format(self.dim, self.regards_dim, base.shape)
            )
        if not is_power_of_two(self.base.shape[self.dim]):
            raise ValueError(
                'The swizzled dim {} must be a power of 2, got length {}'.format(self.dim, self.shape[self.dim])
            )
        super().__init__(shape=self.base.shape, size=self.base.size)

    def global2local(self, *args: Int) -> Int:
        assert len(args) == len(self.shape)
        origin_indices = list(args)
        indices = []
        for dim, origin_index in enumerate(origin_indices):
            if dim == self.dim:
                regards_index = origin_indices[self.regards_dim] // (2**self.log_step)
                regards_extent = self.shape[self.regards_dim] // (2**self.log_step)
                if regards_extent > self.shape[dim]:
                    regards_index = regards_index % self.shape[dim]  # prevent the xor making the index out of bound
                indices.append(origin_index ^ regards_index)
            else:
                indices.append(origin_index)
        return self.base.global2local(*indices)

    def global2cond(self, *args: Int) -> Bool:
        return self.base.global2cond(*args)


class TiledDataLayout(DataLayout):
    def __init__(self, base: DataLayout, inner_shape: Sequence[Int]):
        assert len(inner_shape) == len(base.shape)
        assert all(b % a == 0 for a, b in zip(inner_shape, base.shape) if isinstance(a, int) and isinstance(b, int))
        self.base = base
        self.inner_shape = inner_shape
        super().__init__(shape=[b // a for a, b in zip(inner_shape, self.shape)] + list(inner_shape), size=base.size)

    def base_args(self, *args):
        outer_args, inner_args = args[: len(args) // 2], args[len(args) // 2 :]
        return [o * factor + i for factor, o, i in zip(self.inner_shape, outer_args, inner_args)]

    def global2local(self, *args):
        return self.base(*self.base_args(args))

    def global2cond(self, *args):
        return self.base.within_bound(*self.base_args(args))


class SplitDataLayout(DataLayout):
    """
    3-dimension tensor with shape [a, b, c]
    after split(dim2factor={0: 2, 1: 3}) got
    5-dimension tensor with shape [(a + 1) // 2, 2, (b + 2) // 3, 3, c]
    """

    def __init__(self, base: DataLayout, dim2factor: Mapping[int, Int]):
        self.base = base
        self.dim2factor = dim2factor
        shape = []
        for i, s in enumerate(base.shape):
            if i in dim2factor:
                factor = dim2factor[i]
                outer = (s + factor - 1) // factor
                shape.extend([outer, factor])
            else:
                shape.append(s)
        super().__init__(shape=shape, size=base.size)

    def base_args(self, *args):
        merged_args = []
        c = 0
        for i in range(len(self.base.shape)):
            if i in self.dim2factor:
                outer_idx = args[c]
                inner_idx = args[c + 1]
                merged_args.append(outer_idx * self.dim2factor[i] + inner_idx)
                c += 2
            else:
                merged_args.append(args[c])
                c += 1
        return merged_args

    def global2local(self, *args):
        return self.base(*self.base_args(*args))

    def global2cond(self, *args: Int) -> Bool:
        return self.base.within_bound(*self.base_args(*args))


class FusedDataLayout(DataLayout):
    """
    3-dimension tensor with shape [a, b, c]
    after fuse([2, [1, 0]]) got
    3-dimension tensor with shape [c, b * a]
    (i, j, k) of the result data layout will be mapped to (k, j * I + i) of the original data layout
    """

    def __init__(self, base: DataLayout, dim2fuse: Sequence[Union[Sequence[int], int]]):
        self.base = base
        self.dim2fuse = dim2fuse
        covered = []
        shape = []
        self.dims = []
        for i, item in enumerate(dim2fuse):
            if isinstance(item, int):
                item = [item]
            else:
                item = list(item)
            self.dims.append(item)
            covered.extend(item)
            shape.append(prod([base.shape[i] for i in item]))

        msg = "missing some dimension or duplicated dimension"
        assert len(covered) == len(base.shape) and len(set(covered)) == len(covered), msg

        super().__init__(shape=shape, size=base.size)

    def base_args(self, *args: Int):
        original_args = [None] * len(self.base.shape)
        for i in range(len(self.dims)):  # pylint: disable=consider-using-enumerate
            dim_sizes = [self.base.shape[v] for v in self.dims[i]]
            for j, dim in enumerate(self.dims[i]):
                original_args[dim] = args[i] // prod(dim_sizes[j + 1 :]) % dim_sizes[j]
        return original_args

    def global2local(self, *args: Int) -> Int:
        return self.base(*self.base_args(*args))

    def global2cond(self, *args: Int) -> Bool:
        return self.base.within_bound(*self.base_args(*args))


class SliceOutDataLayout(DataLayout):
    """
    3-dimension tensor with shape [a, b, c]
    after cut({0, 2}) got
    1-dimension tensor with shape [b]
    """

    def __init__(self, base: DataLayout, dims: Sequence[int]):
        assert all(d < len(base.shape) for d in dims)
        self.base = base
        self.dims = set(dims)
        super().__init__(
            shape=[s for r, s in enumerate(base.shape) if r not in dims], size=base.size
        )  # todo: update size

    def base_args(self, *args: Int):
        merged_args = []
        c = 0
        for i in range(len(self.base.shape)):
            if i in self.dims:
                merged_args.append(0)
            else:
                merged_args.append(args[c])
                c += 1
        return merged_args

    def global2local(self, *args: Int) -> Int:
        return self.base(*self.base_args(*args))

    def global2cond(self, *args: Int) -> Bool:
        return self.base.within_bound(*self.base_args(*args))


class ProductDataLayout(DataLayout):
    def __init__(self, outer: DataLayout, inner: DataLayout):
        assert len(outer.shape) == len(inner.shape)
        super().__init__(shape=[a * b for a, b in zip(outer.shape, inner.shape)], size=outer.size * inner.size)
        self.outer = outer
        self.inner = inner

    def global2local(self, *args: Int) -> Int:
        outer_args = [v // b for v, b in zip(args, self.inner.shape)]
        inner_args = [v % b for v, b in zip(args, self.inner.shape)]
        return self.outer(*outer_args) * self.inner.size + self.inner(*inner_args)

    def global2cond(self, *args: Int) -> Bool:
        from hidet.ir.expr import LogicalAnd

        outer_args = [v // b for v, b in zip(args, self.inner.shape)]
        inner_args = [v % b for v, b in zip(args, self.inner.shape)]
        return LogicalAnd(self.outer.within_bound(*outer_args), self.inner.within_bound(*inner_args))


class ConcatDataLayout(DataLayout):
    def __init__(self, lhs: DataLayout, rhs: DataLayout):
        super().__init__(shape=list(lhs.shape) + list(rhs.shape), size=lhs.size * rhs.size)
        self.lhs = lhs
        self.rhs = rhs

    def global2local(self, *args: Int) -> Int:
        lhs_args = args[: len(self.lhs.shape)]
        rhs_args = args[len(self.lhs.shape) :]
        return self.lhs(*lhs_args) * self.rhs.size + self.rhs(*rhs_args)

    def global2cond(self, *args: Int) -> Bool:
        from hidet.ir.expr import LogicalAnd

        lhs_args = args[: len(self.lhs.shape)]
        rhs_args = args[len(self.lhs.shape) :]
        return LogicalAnd(self.lhs.within_bound(*lhs_args), self.rhs.within_bound(*rhs_args))


def row_layout(*shape: int):
    return DataLayout.row_major(shape)


def col_layout(*shape: int):
    return DataLayout.column_major(shape)


def local_layout(*shape: int):
    return DataLayout.local(shape)


def data_layout(shape: List[int], ranks: Optional[List[int]] = None):
    if ranks is None:
        ranks = list(range(len(shape)))
    return StridesLayout.from_shape(shape, ranks)
