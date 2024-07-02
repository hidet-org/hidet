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

###################################################################################################
# The following code is inspired by the core component, CuTe, in CUTLASS 3.5:
# https://github.com/NVIDIA/cutlass/blob/main/python/pycute/int_tuple.py
##################################################################################################/
# This file is a python implementation for utilities in CuTe, which will be
# used for integrating CuTe dialect.
from typing import Union
from hidet.ir.expr import Expr, Constant, is_constant, if_then_else


def is_integer(i):
    return isinstance(i, (int, Expr))


def constant_value(a: Union[Constant, int]):
    return a.value if isinstance(a, Constant) else a


def repeat_like(a, val: int = 0):
    if is_integer(a):
        return val
    else:
        assert isinstance(a, tuple)
        return tuple(repeat_like(i, val) for i in a)


def unflatten(a, profile: int):
    if isinstance(profile, int):
        assert isinstance(a, int)
        return a
    else:
        assert len(flatten(profile)) == len(a)
        left = right = 0
        ret = []
        for i in profile:
            if is_integer(i):
                right += 1
                ret.append(a[left])
            else:
                right += len(flatten(i))
                ret.append(unflatten(a[left:right], i))
            left = right
        return tuple(ret)


def signum(a):
    assert is_integer(a)
    if isinstance(a, Expr):
        return 1
    else:
        return (0 < a) - (a < 0)


def depth(a):
    if isinstance(a, tuple):
        d = 0
        for i in a:
            d = max(depth(i), d)
        return d + 1
    else:
        assert is_integer(a)
        return 0


def flatten(a):
    if is_integer(a):
        return a
    else:
        assert isinstance(a, tuple)
        if len(a) == 1:
            if depth(a) > 1:
                return flatten(a[0])
            else:
                return a
        else:
            ret = tuple()
            for i in a:
                if is_integer(i):
                    ret += tuple([i])
                else:
                    assert isinstance(i, tuple)
                    ret += flatten(i)
            return ret


def rank(a):
    if is_integer(a):
        return 1
    else:
        assert isinstance(a, tuple)
        return len(a)


def product(a):
    if is_integer(a):
        return a
    else:
        ret = 1
        for i in a:
            ret *= product(i)
        return ret


def product_each(a):
    return (product(i) for i in a)


def prefix_product(a, init: int = 1):
    if is_integer(a):
        return init
    else:
        from itertools import accumulate

        def mul(a: Union[Expr, int], b: Union[Expr, int]):
            if isinstance(a, Expr) or isinstance(b, Expr):
                return a * b
            else:
                assert isinstance(a, int) and isinstance(b, int)
                return a * b

        return unflatten(tuple(accumulate(tuple([init]) + flatten(a)[:-1], mul)), a)


def size(a):
    if isinstance(a, tuple):
        return product(a)
    else:
        assert is_integer(a)
        return a


def inner_product(a, b):
    if isinstance(a, tuple):
        assert isinstance(b, tuple)
        assert compatible(a, b)
        dot = 0
        for x, y in zip(a, b):
            dot += inner_product(x, y)
        return dot
    else:
        assert isinstance(a, int) and isinstance(b, int)
        return a * b


def ceil_div(a, b):
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        return tuple(ceil_div(x, y) for x, y in zip(a, b))
    else:
        assert is_integer(a) and is_integer(b)
        if not is_constant(b) or b != 0:
            return (a + b - 1) // b
        else:
            assert a == 0
            return 1


def shape_min(a, b):
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        return (shape_min(x, y) for x, y in zip(a, b))
    else:
        assert is_integer(a)
        if isinstance(b, tuple):
            return shape_min(a, product(b))
        else:
            assert is_integer(b)
            if any(isinstance(v, Expr) for v in [a, b]):
                return if_then_else(a > b, b, a)
            else:
                return min(a, b)


def shape_abs(a):
    if isinstance(a, tuple):
        return tuple(shape_abs(x) for x in a)
    else:
        assert is_integer(a)
        if not is_constant(a):
            return if_then_else(a > 0, a, -a)
        else:
            a = constant_value(a)
            return abs(a)


def shape_div(a, b):
    if isinstance(a, tuple):
        if isinstance(b, tuple):
            assert len(a) == len(b)
            return (shape_div(x, y) for x, y in zip(a, b))
        else:
            r = []
            for v in a:
                r.append(shape_div(v, b))
                b = shape_div(b, product(v))
            return tuple(r)
    else:
        assert is_integer(a)
        if isinstance(b, tuple):
            return shape_div(a, product(b))
        else:
            assert is_integer(b)
            if any(isinstance(v, Expr) for v in [a, b]):
                return if_then_else(a // b != 0, a // b, signum(a) * signum(b))
            else:
                # commented out, re-enable after we fix all the testcases
                # assert a % b == 0 or b % a == 0
                return a // b if a // b != 0 else signum(a) * signum(b)


def elem_scale(a, b):
    if isinstance(a, tuple):
        return (elem_scale(x, y) for x, y in zip(a, b))
    else:
        return a * product(b)


def congruent(a, b):
    """Test if two tuple have the same profile (hierarchical rank division)"""
    return repeat_like(a) == repeat_like(b)


def compatible(a, b):
    """
    Test if Shape B is compatible with Shape A:
    Any coordinate into A can also be used as a coordiante into B
    A <= B is a partially ordered set of factored shapes
    """
    if (isinstance(a, tuple)) and (isinstance(b, tuple)):
        if len(a) != len(b):
            return False
        else:
            compat = True
            for x, y in zip(a, b):
                compat &= compatible(x, y)
            return compat
    elif isinstance(a, int):
        return a == size(b)
    else:
        assert isinstance(b, int)
        return False


def crd2idx(coord, shape, stride=None):
    """
    Takes the coordinates, shape, and stride as input and return the offset (inner
    product between coordinates and strides). The coordinates can be 1-dimensional,
    multi-dimensional or hierarchical, and must be compatible with the shape. 1-d
    coordinate will be converted to coordinates that is compatible with the shape,
    and then perform the inner product.
    """
    if stride is None:
        if isinstance(shape, tuple):
            assert len(coord) == len(shape)
            # Horner's method to compute the offset
            flat_coord = flatten(coord)
            flat_shape = flatten(shape)
            ret = flat_coord[0]
            for c, s in zip(flat_coord[1:], flat_shape[1:]):
                ret += c + s * ret
            return ret
        else:
            return coord
    else:
        if isinstance(coord, tuple):
            if isinstance(shape, tuple):  # tuple tuple tuple
                assert len(coord) == len(shape), "Mismatched Ranks"
                assert len(coord) == len(stride), "Mismatched Ranks"
                ret = 0
                for c, s, d in zip(coord, shape, stride):
                    ret += crd2idx(c, s, d)
                return ret
            else:
                raise ValueError(f"Incompatible coordinates and shape.(coord:{coord},shape:{shape})")
        else:
            if isinstance(shape, tuple):
                assert isinstance(stride, tuple), "Invalid Parameters"
                assert len(shape) == len(stride), "Mismatched Ranks"
                ret = 0
                for s, d in zip(shape, stride):
                    ret += crd2idx(coord % product(s), s, d)
                    coord = coord // product(s)
                return ret
            else:
                assert is_integer(stride)
                return coord * stride


def compact_col_major(shape, current: int = 1):
    return prefix_product(shape, current)


def compact_row_major(shape, current: int = 1):
    if is_integer(shape):
        return current
    else:
        from itertools import accumulate

        def mul(a: Union[Expr, int], b: Union[Expr, int]):
            if isinstance(a, Expr) or isinstance(b, Expr):
                return a * b
            else:
                assert isinstance(a, int) and isinstance(b, int)
                return a * b

        a = reversed(flatten(shape)[1:] + tuple([current]))
        return unflatten(tuple(reversed(list(accumulate(a, mul)))), shape)


# TODO: This function seems to have the same functionality as
# index_deserialize. Should consider how to extend to support Expr as intputs
# and how to converge them.
def idx2crd(idx, shape, stride=None):
    """
    a shortcut when we know the Stride is default [CompactColMajor]
    (idx / 1) % s0
    (idx / s0) % s1
    (idx / (s0 * s1)) % s2
    ...
    """
    if stride is None:
        if isinstance(idx, tuple):
            if isinstance(shape, tuple):  # tuple, tuple
                assert len(idx) == len(shape), "Mismatched Ranks"
                return tuple(idx2crd(i, s) for i, s in zip(idx, shape))
            else:
                assert False, "Invalid Parameters"
        else:
            if isinstance(shape, tuple):  # int, tuple
                return idx2crd(idx, shape, compact_col_major(shape))
            else:
                return idx
    else:
        if isinstance(idx, tuple):
            if isinstance(shape, tuple):  # tuple tuple tuple
                assert len(idx) == len(shape), "Mismatched Ranks"
                assert len(idx) == len(stride), "Mismatched Ranks"
                return tuple(idx2crd(i, s, d) for i, s, d in zip(idx, shape, stride))
            else:
                assert False, "Invalid Parameters"
        else:
            if isinstance(shape, tuple):
                if isinstance(stride, tuple):  # int tuple tuple
                    assert len(shape) == len(stride), "Mismatched Ranks"
                    return tuple(idx2crd(idx, s, d) for s, d in zip(shape, stride))
                else:  # int tuple int
                    return tuple(idx2crd(idx, s, d) for s, d in zip(shape, compact_col_major(shape, stride)))
            else:  # int int int
                if shape == 1:
                    return 0
                else:
                    return (idx // stride) % shape


def filter_zeros(a, b):
    """
    Replace the elements of Tuple b that are paired with an 0-stride with an
    1-shape
    """
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(b) == len(a)
        return tuple(filter_zeros(x, y) for x, y in zip(a, b))
    else:
        assert is_integer(a)
        return 1 if is_constant(a) and a == 0 else b


def is_static(a):
    if isinstance(a, int):
        return True
    elif isinstance(a, Expr):
        return is_constant(a)
    else:
        assert isinstance(a, tuple)
        return all(is_static(v) for v in a)
