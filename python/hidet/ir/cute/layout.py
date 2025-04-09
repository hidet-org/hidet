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
# The following code is inspired by the core component, CuTe, from CUTLASS 3.5:
# https://github.com/NVIDIA/cutlass/blob/main/python/pycute/layout.py
##################################################################################################/
# This file is a python implementation for the layout (core concept) in CuTe, which will be
# used for integrating CuTe dialect.
from typing import List, Tuple, Union, Optional, Callable
import enum
from enum import auto as enum_auto

from hidet.ir.expr import Expr, var, is_constant
from .int_tuple import (
    repeat_like,
    signum,
    flatten,
    prefix_product,
    size,
    ceil_div,
    shape_div,
    elem_scale,
    congruent,
    crd2idx,
    compact_col_major,
    idx2crd,
    filter_zeros,
    is_integer,
    shape_abs,
    shape_min,
    is_static,
    canonicalize_uni_shape,
    is_tuple,
    slice_,
    has_none,
    shape_eq,
    product,
)
from .swizzle import Swizzle


def tuple_(a: Union[None, list, tuple, int]):
    if a is None or is_integer(a):
        return a
    else:
        assert isinstance(a, (list, tuple))
        return tuple(tuple_(x) for x in a)


class LayoutBase:
    pass


# cute layout
# TODO: choose a better name
class TensorLayout(LayoutBase):
    def __init__(self, shape, stride=None):
        shape = tuple_(shape)
        stride = tuple_(stride)
        self.shape = shape
        if stride is None:
            stride = prefix_product(shape)
        assert congruent(shape, stride)
        self.stride = stride

    @property
    def shape_tuple(self) -> Tuple[Union[int, Expr], ...]:
        if is_integer(self.shape):
            return tuple([self.shape])
        else:
            return self.shape

    @property
    def stride_tuple(self) -> Tuple[Union[int, Expr], ...]:
        if is_integer(self.stride):
            return tuple([self.stride])
        else:
            return self.stride

    def __str__(self):
        return f"{self.shape}:{self.stride}"

    def __getitem__(self, i):
        if is_integer(self.stride):
            return TensorLayout(self.shape, self.stride)
        else:
            return TensorLayout(self.shape[i], self.stride[i])

    def __call__(self, i, base: Optional[Union[int, Expr]] = None) -> Expr:
        if isinstance(i, list):
            i = tuple(i)

        if has_none(i):
            assert base is None
            if len(i) == 1:
                return TensorLayout(slice_(i[0], self.shape), slice_(i[0], self.stride))
            else:
                return TensorLayout(slice_(i, self.shape), slice_(i, self.stride))
        else:
            if is_integer(i):
                crd = idx2crd(i, self.shape)
            else:
                assert is_tuple(i)
                crd = i
            assert (is_integer(crd) and is_integer(self.shape)) or len(crd) == len(self.shape)
            idx = crd2idx(crd, self.shape, self.stride)
            return base + idx if base is not None else idx

    def __eq__(self, other):
        return shape_eq(self.shape, other.shape) and shape_eq(self.stride, other.stride)

    def size(self):
        return size(self.shape)

    def cosize(self):
        flat_shape = flatten(self.shape)
        flat_stride = flatten(self.stride)
        if is_integer(flat_stride):
            return (flat_shape - 1) * shape_abs(flat_stride) + 1
        else:
            abs_sub_layout = TensorLayout(flat_shape, tuple(shape_abs(i) for i in flat_stride))
            return abs_sub_layout(abs_sub_layout.size() - 1) + 1

    def depth(self):
        from .int_tuple import depth

        return depth(self.shape)

    def reversed(self):
        return TensorLayout(tuple(reversed(self.shape)), tuple(reversed(self.stride)))

    def compose(self, other):
        return composition(self, other)

    def count(self):
        return filter(self).size()


# TODO: rename to something like DynamicLayout
class AutoLayout(TensorLayout):
    def __init__(self, shape, stride=None):
        shape = tuple_(shape)
        if shape is not None and stride is None:
            stride = repeat_like(shape, var('v'))
        super().__init__(shape, stride)

    def __str__(self):
        if self is auto_layout:
            return "auto_layout"
        return f"{self.shape}:{self.stride}"

    def __getitem__(self, i):
        if self is auto_layout:
            raise NotImplementedError(f"Cannot get item from {self}")
        if is_integer(self.stride):
            return AutoLayout(self.shape, self.stride)
        else:
            return AutoLayout(self.shape[i], self.stride[i])

    def __call__(self, i, base: Optional[Union[int, Expr]] = None) -> Expr:
        raise NotImplementedError()

    def size(self):
        if self is auto_layout:
            raise NotImplementedError(f"Cannot get size of {self}")
        return size(self.shape)

    def cosize(self):
        raise NotImplementedError()

    def depth(self):
        from .int_tuple import depth

        if self is auto_layout:
            raise NotImplementedError(f"Cannot get depth of {self}")
        return depth(self.shape)

    def reversed(self):
        if self is auto_layout:
            raise NotImplementedError(f"Cannot reverse {self}")
        return AutoLayout(tuple(reversed(self.shape)), tuple(reversed(self.stride)))

    def compose(self, other):
        raise NotImplementedError()

    def count(self):
        raise NotImplementedError()


def layout_auto(shape: Tuple[int, ...], stride: Optional[Tuple[int, ...]] = None):
    return AutoLayout(shape, stride)


def is_auto_layout(layout: LayoutBase):
    return isinstance(layout, AutoLayout)


class AutoLayoutSingleton(AutoLayout):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(AutoLayoutSingleton, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        super().__init__(None)


auto_layout = AutoLayoutSingleton()


def commutative(functor: Callable[Expr, Expr], layout: TensorLayout):
    if not isinstance(functor, Swizzle):
        return False
    yyy_bits = functor.base + max(0, functor.shift)
    zzz_bits = functor.base - min(0, functor.shift)
    yyy_hi = 1 << (functor.bits + yyy_bits)
    yyy_lo = 1 << (yyy_bits - 1)
    zzz_hi = 1 << (functor.bits + zzz_bits)
    zzz_lo = 1 << (zzz_bits - 1)
    layout = filter(layout)
    shapes = flatten(layout.shape_tuple)
    strides = flatten(layout.stride_tuple)

    def cond(s: int, d: int):
        return d % yyy_hi == 0 or (s * d <= yyy_lo and d % zzz_hi == 0) or s * d <= zzz_lo

    return all(cond(s, d) for s, d in zip(shapes, strides))


class ComposedTensorLayout(LayoutBase):
    def __init__(self, layout: TensorLayout, base: Union[Expr, int], functor: Callable):
        self.layout = layout
        self.base = base
        self.functor = functor

    @property
    def shape(self):
        return self.layout.shape

    @property
    def stride(self):
        return self.layout.stride

    @property
    def shape_tuple(self) -> Tuple[Union[int, Expr], ...]:
        return self.layout.shape_tuple

    @property
    def stride_tuple(self) -> Tuple[Union[int, Expr], ...]:
        return self.layout.stride_tuple

    def to_tensor_layout(self) -> TensorLayout:
        if isinstance(self.functor, TensorLayout):
            return composition(self.functor, self.layout)
        else:
            return self.layout

    def __str__(self):
        return f"layout:{self.layout},base:{self.base},functor:{self.functor}"

    def __getitem__(self, i):
        return ComposedTensorLayout(self.layout[i], self.base, self.functor)

    def __call__(self, i, base: Optional[Union[int, Expr]] = None) -> Expr:
        if has_none(i):
            assert base is None
            return ComposedTensorLayout(self.layout(i), self.base, self.functor)
        else:
            b = self.base + base if base is not None else self.base
            if commutative(self.functor, self.layout):
                return self.functor(b) + self.layout(i)
            else:
                return self.functor(b + self.layout(i))

    def __eq__(self, other):
        raise NotImplementedError

    def size(self):
        return self.layout.size()

    def cosize(self):
        return self.functor.cosize()

    def depth(self):
        return self.layout.depth()

    def reversed(self):
        return NotImplementedError

    def compose(self, other: TensorLayout):
        return ComposedTensorLayout(composition(self.layout, other), self.base, self.functor)

    def count(self):
        return self.layout.count()


def make_layout(*layouts):
    result = TensorLayout(tuple(layout.shape for layout in layouts), tuple(layout.stride for layout in layouts))

    assert len(layouts) >= 1
    layout = layouts[0]
    if isinstance(layout, TensorLayout):
        return result
    else:
        assert isinstance(layout, ComposedTensorLayout)
        return ComposedTensorLayout(result, layout.base, layout.functor)


def coalesce(a: Union[TensorLayout, ComposedTensorLayout]):
    """
    Coalesce first flattens the layout, removes dummy modes and then merges
    some contiguous modes into a single mode. This operation simplify the
    layout representation.
    For example.
    a = TensorLayout((2, (1, 6)), (1, (6, 2)))
    b = coalesce(a) # 12:1
    For detailed documentation, please refer to
    https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#coalesce
    """
    if isinstance(a, ComposedTensorLayout):
        coalesced = coalesce(a.layout)
        return ComposedTensorLayout(coalesced, a.base, a.functor)

    if is_integer(a.shape):
        return a if (not is_constant(a.shape) or a.shape != 1) else TensorLayout(1)
    else:
        flat_shape = flatten(a.shape)
        flat_stride = flatten(a.stride)
        result_shape = []
        result_stride = []
        for s, d in zip(flat_shape, flat_stride):
            if len(result_shape) == 0:
                if not is_constant(s) or s != 1:
                    result_shape.append(s)
                    result_stride.append(d)
            else:
                curr_shape = result_shape[-1]
                curr_stride = result_stride[-1]
                if all(is_constant(e) for e in [d, curr_shape, curr_stride]) and d == curr_shape * curr_stride:
                    result_shape[-1] = curr_shape * s
                elif not is_constant(s) or s != 1:
                    result_shape.append(s)
                    result_stride.append(d)
        if len(result_shape) == 0:
            return TensorLayout(1)
        elif len(result_shape) == 1:
            return TensorLayout(result_shape[0], result_stride[0])
        else:
            return TensorLayout(tuple(result_shape), tuple(result_stride))


def filter(a: TensorLayout, if_coalesce: bool = True):
    b = TensorLayout(filter_zeros(a.stride, a.shape), a.stride)
    if if_coalesce:
        return coalesce(b)
    else:
        return b


def composition(a: Union[TensorLayout, ComposedTensorLayout], b: TensorLayout):
    """
    Composition of two layouts: lhs o rhs
    @post compatible(rhs, result)
    @post result(c) = lhs(rhs(c))
            for all c in the domain of rhs
    For detailed documentation, please refer to
    https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#composition
    """
    if isinstance(a, ComposedTensorLayout):
        a = a.layout

    if isinstance(b.stride, tuple):
        return make_layout(*[composition(a, i) for i in b])
    else:
        assert is_integer(b.stride)

        flat_shape = flatten(a.shape)
        flat_stride = flatten(a.stride)
        if b.stride == 0:
            return TensorLayout(b.shape, b.stride)
        elif is_integer(a.shape):
            result_stride = b.stride * a.stride
            return TensorLayout(b.shape, result_stride)
        elif b.stride == 1:
            result_shape = []
            rest_shape = b.shape
            for s in flat_shape[:-1]:
                result_shape.append(shape_min(shape_abs(s), rest_shape))
                rest_shape = shape_div(rest_shape, shape_abs(s))
            result_shape.append(rest_shape)
            return coalesce(TensorLayout(tuple(result_shape), tuple(flat_stride)))
        else:
            rest_shape = b.shape
            rest_stride = b.stride
            result_shape = []
            result_stride = []
            for s, d in zip(flat_shape[:-1], flat_stride[:-1]):
                s1 = shape_div(s, rest_stride)
                rest_stride = shape_div(rest_stride, s)
                d1 = elem_scale(d, shape_div(s, s1))
                s2 = shape_min(shape_abs(s1), rest_shape)
                rest_shape = shape_div(rest_shape, shape_abs(s1))
                result_shape.append(s2)
                result_stride.append(d1)
            result_shape.append(rest_shape)
            result_stride.append(rest_stride * flat_stride[-1])
            return coalesce(TensorLayout(tuple(result_shape), tuple(result_stride)))


def complement(a: TensorLayout, cosize_hi: int = None):
    """
    Complement

    Build the complement of a layout.
    @post size(@a result) >= @a cosize_hi / size(filter(@a layout)));
    @post For all i in [1,size(@a result)),
              @a result(i) < @a result(i-1)
              For all j in [0, size(@a layout)),
                  @a result(i) != @a layout(j)
    For detailed documentation, please refer to
    https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#complement
    """
    if cosize_hi is None:
        cosize_hi = a.cosize()
    filter_layout = filter(a)
    filter_shape = filter_layout.shape
    filter_stride = filter_layout.stride
    if is_integer(filter_stride) and is_constant(filter_stride) and filter_stride == 0:
        return TensorLayout(cosize_hi)
    else:
        if is_integer(filter_shape):
            filter_shape = [filter_shape]
            filter_stride = [filter_stride]
        result_shape = []
        result_stride = [1]
        sorted_DS = sorted(zip(filter_stride, filter_shape))
        for d, s in sorted_DS[:-1]:
            result_shape.append(d // result_stride[-1])
            result_stride.append(s * d)
        last_stride, last_shape = sorted_DS[-1]
        result_shape.append(last_stride // result_stride[-1])
        rest_stride = last_shape * last_stride
        result_shape.append(ceil_div(cosize_hi, rest_stride))
        result_stride.append(rest_stride)
        return coalesce(TensorLayout(tuple(result_shape), tuple(result_stride)))


def right_inverse(a: TensorLayout):
    """
    Build the right-inverse of a layout
    @pre is_static<Layout>
    @result A layout @a result such that
       @a layout(@a result(i)) == i for all i < size(@a result)
    @result A layout @a result such that
       composition(@a layout, @a result) is identical to make_layout(shape(result))
    """
    flat_layout = coalesce(a)
    if is_integer(flat_layout.stride):
        flat_shape = tuple([flat_layout.shape])
        flat_stride = [flat_layout.stride]
    else:
        flat_shape = flat_layout.shape
        flat_stride = [shape_abs(i) for i in flat_layout.stride]
    result_shape = []
    result_stride = []
    current_idx = 1
    for d, s, rstride in sorted(zip(flat_stride, flat_shape, compact_col_major(flat_shape))):
        if d != current_idx:
            break
        result_shape.append(s)
        result_stride.append(signum(d) * rstride)
        current_idx = s * d
    if len(result_stride) == 0:
        return TensorLayout(1, 0)
    return TensorLayout(tuple(result_shape), tuple(result_stride))


def left_inverse(a: TensorLayout):
    """
    Build the left-inverse of a layout
    @pre is_static<Layout>
    @pre @a layout is an injective function
    @result A layout @a result such that
       @a result(@a layout(i)) == i for all i < size(@a layout)
    @result A layout @a result such that
       composition(@a result, @a layout) is identical to make_layout(shape(layout))
    """
    return right_inverse(make_layout(a, complement(a)))


def right_inverse_ignore_zero_strides(a: TensorLayout):
    """
    TODO: add document
    """
    flat_layout = coalesce(a)
    if is_integer(flat_layout.stride):
        flat_shape = tuple([flat_layout.shape])
        flat_stride = [flat_layout.stride]
    else:
        flat_shape = flat_layout.shape
        flat_stride = [shape_abs(i) for i in flat_layout.stride]
    result_shape = []
    result_stride = []
    current_idx = 1
    for d, s, rstride in sorted(zip(flat_stride, flat_shape, compact_col_major(flat_shape))):
        if d == 0:
            continue
        if d != current_idx:
            break
        result_shape.append(s)
        result_stride.append(signum(d) * rstride)
        current_idx = s * d
    if len(result_stride) == 0:
        return TensorLayout(1, 0)
    return TensorLayout(tuple(result_shape), tuple(result_stride))


def left_inverse_ignore_zero_strides(a: TensorLayout):
    """
    TODO: add document
    """
    return right_inverse_ignore_zero_strides(make_layout(a, complement(a)))


def logical_product(a: TensorLayout, b: TensorLayout):
    return make_layout(a, composition(complement(a, a.size() * b.cosize()), b))


def logical_divide(a: TensorLayout, b: TensorLayout):
    return composition(a, make_layout(b, complement(b, a.size())))


def max_common_vector(a: Union[TensorLayout, ComposedTensorLayout], b: Union[TensorLayout, ComposedTensorLayout]):
    if isinstance(a, ComposedTensorLayout):
        a = a.to_tensor_layout()
    if isinstance(b, ComposedTensorLayout):
        b = b.to_tensor_layout()
    if is_static(a.shape) and is_static(a.stride) and is_static(b.shape) and is_static(b.stride):
        common = coalesce(composition(filter(a, False), right_inverse(filter(b, False))))
        shape, stride = common.shape_tuple[0], common.stride_tuple[0]
        if stride == 1:
            return shape
        else:
            return 1
    else:
        return 1


def slice_and_offset(crd: tuple, layout: TensorLayout):
    return (
        TensorLayout(slice_(crd, layout.shape), slice_(crd, layout.stride)),
        crd2idx(crd, layout.shape, layout.stride),
    )


def common_reshape(a: TensorLayout, b: TensorLayout):
    """
    This function reorganizes shapes of the two layouts into congruent form.
    This function assuems the shapes of the two layouts should be flattened.
    For example, we have two layouts
    a := (4, 8, 8):(1, 16, 256)
    b := (16, 4, 4):(1, 32, 512)
    the results of the reshape become
    a_reshape := (4, 4, 2, 2, 4):(1, 16, 32, 256, 512)
    b_reshape := (4, 4, 2, 2, 4):(1, 4, 32, 64, 512)
    This operation is useful when we perform elementwise operations on two
    tensors. After we apply this operation, the tensors will have the aligned
    shape, so that we can perform elementwise on them.
    """
    result_shape_a = []
    result_shape_b = []
    result_stride_a = []
    result_stride_b = []

    shape_a = list(a.shape_tuple)
    shape_b = list(b.shape_tuple)
    stride_a = list(a.stride_tuple)
    stride_b = list(b.stride_tuple)

    while len(shape_a) > 0 and len(shape_b) > 0:
        sa = shape_a[0]
        sb = shape_b[0]
        result_stride_a.append(stride_a[0])
        result_stride_b.append(stride_b[0])
        if sa > sb:
            assert sa % sb == 0
            s = sb
            shape_a[0] //= sb
            stride_a[0] *= sb
            _, _ = shape_b.pop(0), stride_b.pop(0)
        elif sb > sa:
            assert sb % sa == 0
            s = sa
            shape_b[0] //= sa
            stride_b[0] *= sa
            _, _ = shape_a.pop(0), stride_a.pop(0)
        else:
            s = sa
            _, _ = shape_a.pop(0), stride_a.pop(0)
            _, _ = shape_b.pop(0), stride_b.pop(0)
        result_shape_a.append(s)
        result_shape_b.append(s)
    return TensorLayout(tuple(result_shape_a), tuple(result_stride_a)), TensorLayout(
        tuple(result_shape_b), tuple(result_stride_b)
    )


def group(layout: Union[TensorLayout, ComposedTensorLayout], size_: int, filter_zero: bool = False):
    """
    Split a layout into two parts such that the size of first part equals the provided size_

    Parameters:
        layout (Union[TensorLayout, ComposedTensorLayout]): The layout to be split.
        size_ (int): The size of the first part.
        filter_zero (bool): Whether to filter the zero strides.

    Returns:
        Tuple[TensorLayout, TensorLayout]: The first part and the second part of the layout.

    Example:
        Suppose we have a layout which is
        ```
        layout = ((2, 2), 8):((0, 1), 2)
        ```
        and the size of the first part is 2. Then, we can split the layout into two parts
        ```
        a = (2, ): (0, )
        b = (2, 8): (1, 2)
        ```
        If filter_zero is True, the result will be
        ```
        a = (2, 2): (0, 1) # skip the zero strides
        b = (8, ): (2, )
        ```
    """
    flat_shape = list(flatten(layout.shape_tuple))
    flat_stride = list(flatten(layout.stride_tuple))
    result_shape = []
    result_stride = []
    rest_shape = []
    rest_stride = []
    current_idx = 1
    rest = False
    for s, d in zip(flat_shape, flat_stride):
        if filter_zero and d == 0:
            if rest:
                rest_shape.append(s)
                rest_stride.append(d)
            else:
                result_shape.append(s)
                result_stride.append(d)
        elif current_idx * s <= size_:
            result_shape.append(s)
            result_stride.append(d)
            current_idx *= s
            if current_idx == size_:
                rest = True
        elif current_idx * s > size_:
            if not rest:
                shape = shape_div(size_, current_idx)
                # same as below
                if s % shape != 0:
                    return None, None
                remaining = shape_div(s, shape)
                result_shape.append(shape)
                result_stride.append(d)
                rest_shape.append(remaining)
                rest_stride.append(d * shape)
                rest = True
            else:
                rest_shape.append(s)
                rest_stride.append(d)
            current_idx *= s
        # Note: this check is for non-power-two tiles.
        # non-power-two tiles may cause indivisible error during
        # instruction selection pass. Therefore, we should do sanity
        # check here to reject invalid instruction.
        if current_idx < size_ and size_ % current_idx != 0:
            return None, None

    def tensor_layout(shape, stride):
        if len(shape) > 1:
            return TensorLayout(tuple(shape), tuple(stride))
        elif len(shape) == 1:
            return TensorLayout(shape[0], stride[0])
        else:
            return TensorLayout(1)

    result = tensor_layout(result_shape, result_stride)
    rest = tensor_layout(rest_shape, rest_stride)
    if isinstance(layout, TensorLayout):
        return result, rest
    else:
        assert isinstance(layout, ComposedTensorLayout)
        return ComposedTensorLayout(result, layout.base, layout.functor), ComposedTensorLayout(
            rest, layout.base, layout.functor
        )


def remove_lo_hi(layout: TensorLayout, lo: int, hi: int):
    """
    Remove the modes with strides greeater than lo and less than hi. This function
    is used when checking if the input and output layouts are compatible for a
    reduction operation.

    Parameters:
        layout (TensorLayout): The layout to be processed.
        lo (int): The lower bound of the strides.
        hi (int): The upper bound of the strides.

    Returns:
        TensorLayout: The layout after removing the modes with strides greater
         than lo and less than hi.

    Example:
        Suppose we have a tensor with thread-value layout
        ```
        tv = ((4, 8), (2, 2)): ((32, 1), (16, 8))
        ```
        and the shape of this layout is (16, 8). If we have a reduce operation
        that reduces the tensor along the first dimension, the output thread-value
        layout should be
        ```
        tv = ((4, 8), (2, 2)): ((32, 0), (16, 0))
        ```
        Then, the lower bound of the strides is 1 and the upper bound of the strides
        is 16 in the sense of column-major order. So, we use this function to remove
        the strides that is greater than 1 and less than 16.
        >>> remove_lo_hi(tv, 1, 16)
        ((4, ), (2, )): ((32, ), (16, ))
    """
    shape = flatten(layout.shape_tuple)
    stride = flatten(layout.stride_tuple)
    result_shape = []
    result_stride = []
    for s, d in zip(shape, stride):
        if d < lo:
            if s * d > lo:
                s1 = shape_div(lo, d)
                result_shape.append(s1)
                result_stride.append(d)
                s2 = shape_div(s, s1)
                if s2 * lo > hi:
                    s3 = shape_div(hi, lo)
                    s4 = shape_div(s2, s3)
                    result_shape.append(s4)
                    result_stride.append(hi)
            else:
                result_shape.append(s)
                result_stride.append(d)
        elif lo <= d < hi:
            if s * d > hi:
                s1 = shape_div(hi, lo)
                s2 = shape_div(s, s1)
                result_shape.append(s2)
                result_stride.append(hi)
        else:
            result_shape.append(s)
            result_stride.append(d)
    return TensorLayout(tuple(result_shape), tuple(result_stride))


def filter_lo_hi(layout: TensorLayout, lo: int, hi: int):
    """
    If the strides of the modes are greater than lo and less than hi, replace the
    strides with zeros. This function is used to infer the output layout of the
    reduce operation.

    Parameters:
        layout (TensorLayout): The layout to be processed.
        lo (int): The lower bound of the strides.
        hi (int): The upper bound of the strides.

    Returns:
        TensorLayout: The layout after filtering the modes with strides greater

    Example:
        Suppose we have a tensor with thread-value layout
        ```
        tv = ((4, 8), (2, 2)): ((32, 1), (16, 8))
        ```
        and the shape of this layout is (16, 8). If we have a reduce operation
        that reduces the tensor along the first dimension, the output thread-value
        layout should be
        ```
        tv = ((4, 8), (2, 2)): ((32, 0), (16, 0))
        ```
        Then, the lower bound of the strides is 1 and the upper bound of the strides
        is 16 in the sense of column-major order. So, we use this function to filter
        the strides that is greater than 1 and less than 16.
        >>> filter_lo_hi(tv, 1, 16)
        ((4, 8), (2, 2)): ((32, 0), (16, 0))
    """
    shape = flatten(layout.shape_tuple)
    stride = flatten(layout.stride_tuple)
    result_shape = []
    result_stride = []
    for s, d in zip(shape, stride):
        if d < lo:
            if s * d > lo:
                s1 = shape_div(lo, d)
                result_shape.append(s1)
                result_stride.append(d)
                s2 = shape_div(s, s1)
                if s2 * lo > hi:
                    s3 = shape_div(hi, lo)
                    result_shape.append(s3)
                    result_stride.append(0)
                    s4 = shape_div(s2, s3)
                    result_shape.append(s4)
                    result_stride.append(hi)
                else:
                    result_shape.append(s2)
                    result_stride.append(0)
            else:
                result_shape.append(s)
                result_stride.append(d)
        elif lo <= d < hi:
            if s * d > hi:
                s1 = shape_div(hi, lo)
                s2 = shape_div(s, s1)
                result_shape.append(s1)
                result_stride.append(0)
                result_shape.append(s2)
                result_stride.append(hi)
            else:
                result_shape.append(s)
                result_stride.append(0)
        else:
            result_shape.append(s)
            result_stride.append(d)
    return TensorLayout(tuple(result_shape), tuple(result_stride))


class Label(enum.Enum):
    Thread = enum_auto()
    QuadPair = enum_auto()
    Warp = enum_auto()
    WarpGroup = enum_auto()
    ThreadBlock = enum_auto()
    ThreadBlockCluster = enum_auto()


label_names = {
    Label.Thread: "thread",
    Label.QuadPair: "quad_pair",
    Label.Warp: "warp",
    Label.WarpGroup: "warp_group",
    Label.ThreadBlock: "thread_block",
    Label.ThreadBlockCluster: "thread_block_cluster",
}


name_to_label = {
    "thread": Label.Thread,
    "quad_pair": Label.QuadPair,
    "warp": Label.Warp,
    "warp_group": Label.WarpGroup,
    "thread_block": Label.ThreadBlock,
    "thread_block_cluster": Label.ThreadBlockCluster,
}


class Atom:
    def __init__(
        self,
        level: Union[str, Label],
        shape: Tuple[int, ...],
        repeat_shape: Tuple[int, ...],
        repeat_layout: TensorLayout,
    ):
        """
        Atom is the basic building block of the layout. It usually represents the
        thread-value layout of an instruction or the smallest unit of a thread-value
        layout. It's intended to provide an easy-to-undedrstand interface for the
        users to create the tilling configuration.

        Parameters:
            level (Union[str, Label]): The level of the atom, which can be thread,
                quad_pair, warp, warp_group, thread_block, or thread_block_cluster.
            shape (Tuple[int, ...]): The shape of the atom.
            repeat_shape (Tuple[int, ...]): The shape of the repeat layout.
            repeat_layout (TensorLayout): The repeat layout of the atom.

        Methods:
            repeat_mk: Get the shape and layout of the repeat layout for the m and k
                dimensions.
            repeat_nk: Get the shape and layout of the repeat layout for the n and k
                dimensions.
            str_indented: Get the string representation of the atom with indentation.
        """
        if isinstance(level, str):
            level = name_to_label[level]
        self.level = level
        self.shape = shape
        if repeat_shape is None:
            repeat_shape = repeat_like(self.shape, 1)
        if repeat_layout is None:
            repeat_layout = TensorLayout(repeat_shape)
        self.repeat_shape = repeat_shape
        self.repeat_layout = repeat_layout

    def repeat_mk(self):
        """
        Get the shape and layout of the repeat layout for the m and k dimensions.

        Example:
            Suppose we have an atom that represents an mma instruction with the shape
            (16, 8, 16). Typically, the instruction will be repeated along the m and n
            dimensions. For example, we assume the instrucion is repeated 2 times along
            the n dimension and 2 times along the n dimension. Then the output of this
            function will be
            ```
            shape_mk = (2, 1) # 1 for k and 1 for m
            ```
            We can always assume the repeat in k dimension is 1 because the repeat in
            k dimension can be represented by a loop in the kernel.
            ```
            layout_mk = (2, 1): (1, 2)
            ```
        Note:
            The repeat in m and n cannot be reflected in the kernel code because
            the computation is parallelized across the threads, and the kernel represents
            the code executed by individual hardware units. So we need to explicitly
            specify the repeat in m and n dimensions in the layout.
        """
        from builtins import filter as filter_

        m, _ = self.repeat_shape
        shape_mk = (m, 1)
        shape = flatten(self.repeat_layout.shape_tuple)
        stride = flatten(self.repeat_layout.stride_tuple)
        mode_m = list(filter_(lambda t: t[1] < m, zip(shape, stride)))
        shape = tuple(s for s, _ in mode_m) if len(mode_m) > 0 else 1
        stride = tuple(d for _, d in mode_m) if len(mode_m) > 0 else 0
        return shape_mk, TensorLayout(shape, stride)

    def repeat_nk(self):
        """
        Get the shape and layout of the repeat layout for the n and k dimensions.
        """
        from builtins import filter as filter_

        m, n = self.repeat_shape
        shape_nk = (n, 1)
        shape = flatten(self.repeat_layout.shape_tuple)
        stride = flatten(self.repeat_layout.stride_tuple)
        mode_n = list(filter_(lambda t: t[1] >= m, zip(shape, stride)))
        shape = tuple(s for s, _ in mode_n)
        stride = tuple(d // m for _, d in mode_n)
        return shape_nk, TensorLayout(shape, stride)

    def str_indented(self, depth: int = 0):
        raise NotImplementedError()


class CopyAtom(Atom):
    def __init__(
        self,
        level: Union[str, Label],
        shape: Tuple[int, ...],
        src_thrval_layout: TensorLayout,
        dst_thrval_layout: TensorLayout = None,
        repeat_shape: Tuple[int, ...] = None,
        repeat_layout: TensorLayout = None,
    ):
        super().__init__(level, shape, repeat_shape, repeat_layout)
        if dst_thrval_layout is None:
            dst_thrval_layout = src_thrval_layout
        self.src_thrval_layout = src_thrval_layout
        self.dst_thrval_layout = dst_thrval_layout

    @staticmethod
    def from_tv_atom(atom):
        return CopyAtom(atom.level, atom.shape, atom.layout, atom.layout, atom.repeat_shape, atom.repeat_layout)

    def repeat_mk(self):
        raise NotImplementedError()

    def repeat_nk(self):
        raise NotImplementedError()

    def str_indented(self, depth: int = 0):
        indent = " " * (depth * 2)
        prev_indent = " " * (max(0, depth - 1) * 2)
        return (
            "{"
            + f"\n{indent}level: {label_names[self.level]}, \n{indent}shape: {self.shape}, "
            + f"\n{indent}src: {self.src_thrval_layout}, \n{indent}dst: {self.dst_thrval_layout}"
            + f"\n{indent}repeat_shape: {self.repeat_shape}, \n{indent}repeat_layout: {self.repeat_layout}"
            + f"\n{prev_indent}"
            + "}"
        )


class MmaAtom(Atom):
    def __init__(
        self,
        level: Union[str, Label],
        shape_mnk: Tuple[int, ...],
        a_thrval_layout: TensorLayout,
        b_thrval_layout: TensorLayout,
        c_thrval_layout: TensorLayout,
        d_thrval_layout: TensorLayout = None,
        repeat_shape: Tuple[int, ...] = None,
        repeat_layout: TensorLayout = None,
    ):
        m, n, k = shape_mnk
        shape_mn = (m, n)
        super().__init__(level, shape_mn, repeat_shape, repeat_layout)
        self.m = m
        self.n = n
        self.k = k
        if d_thrval_layout is None:
            d_thrval_layout = c_thrval_layout
        self.a_thrval_layout = a_thrval_layout
        self.b_thrval_layout = b_thrval_layout
        self.c_thrval_layout = c_thrval_layout
        self.d_thrval_layout = d_thrval_layout

    def shape_mn(self):
        return (self.m, self.n)

    def shape_mk(self):
        return (self.m, self.k)

    def shape_nk(self):
        return (self.n, self.k)

    def str_indented(self, depth: int = 0):
        indent = " " * (depth * 2)
        prev_indent = " " * (max(0, depth - 1) * 2)
        return (
            "{"
            + f"\n{indent}level: {label_names[self.level]}, \n{indent}shape: {(self.m, self.n, self.k)}, "
            + f"\n{indent}a: {self.a_thrval_layout}, \n{indent}b: {self.b_thrval_layout}"
            + f"\n{indent}c: {self.c_thrval_layout}\n{indent}repeat_shape: {self.repeat_shape}, "
            + f"\n{indent}repeat_layout: {self.repeat_layout}"
            + f"\n{prev_indent}"
            + "}"
        )


class ThrValAtom(Atom):
    def __init__(
        self,
        level: Union[str, Label],
        shape: Tuple[int, ...],
        layout: TensorLayout,
        repeat_shape: Tuple[int, ...] = None,
        repeat_layout: TensorLayout = None,
    ):
        super().__init__(level, shape, repeat_shape, repeat_layout)
        self.layout = layout

    @staticmethod
    def from_copy_atom_src(copy_atom: CopyAtom):
        return ThrValAtom(
            copy_atom.level,
            copy_atom.shape,
            copy_atom.src_thrval_layout,
            copy_atom.repeat_shape,
            copy_atom.repeat_layout,
        )

    @staticmethod
    def from_copy_atom_dst(copy_atom: CopyAtom):
        return ThrValAtom(
            copy_atom.level,
            copy_atom.shape,
            copy_atom.dst_thrval_layout,
            copy_atom.repeat_shape,
            copy_atom.repeat_layout,
        )

    def repeat_mk(self):
        raise NotImplementedError()

    def repeat_nk(self):
        raise NotImplementedError()

    def str_indented(self, depth: int = 0):
        indent = " " * (depth * 2)
        prev_indent = " " * (max(0, depth - 1) * 2)
        return (
            "{"
            + f"\n{indent}level: {label_names[self.level]}, \n{indent}shape: {self.shape}, "
            + f"\n{indent}layout: {self.layout}, \n{indent}repeat_shape: {self.repeat_shape}, "
            + f"\n{indent}repeat_layout: {self.repeat_layout}"
            + f"\n{prev_indent}"
            + "}"
        )


class Level(Atom):
    def __init__(
        self,
        unit: Union[str, Label],
        level: Union[str, Label],
        shape: Tuple[int, ...],
        layout: TensorLayout,
        repeat_shape: Tuple[int, ...] = None,
        repeat_layout: TensorLayout = None,
    ):
        super().__init__(level, shape, repeat_shape, repeat_layout)
        if isinstance(unit, str):
            unit = name_to_label[unit]
        self.unit = unit
        self.layout = layout

    def level_mk(self):
        m, _ = self.shape
        shape_mk = (m, 1)
        shape = flatten(self.layout.shape_tuple)
        stride = flatten(self.layout.stride_tuple)
        stride = tuple(d if d < m else 0 for d in stride)
        layout_mk = TensorLayout(shape, stride)
        return Level(self.unit, self.level, shape_mk, layout_mk, *self.repeat_mk())

    def level_nk(self):
        m, n = self.shape
        shape_nk = (n, 1)
        shape = flatten(self.layout.shape_tuple)
        stride = flatten(self.layout.stride_tuple)
        stride = tuple(d // m if d >= m else 0 for d in stride)
        layout_nk = TensorLayout(shape, stride)
        return Level(self.unit, self.level, shape_nk, layout_nk, *self.repeat_nk())

    def str_indented(self, depth: int = 0):
        indent = " " * (depth * 2)
        prev_indent = " " * (max(0, depth - 1) * 2)
        return (
            "{"
            + f"\n{indent}unit: {label_names[self.unit]}, \n{indent}level: {label_names[self.level]}, "
            + f"\n{indent}shape: {self.shape}, \n{indent}layout: {self.layout}, "
            + f"\n{indent}repeat_shape: {self.repeat_shape}, \n{indent}repeat_layout: {self.repeat_layout}"
            + f"\n{prev_indent}"
            + "}"
        )


def zoom(atom_shape: Tuple[int, ...], atom: TensorLayout, repeat_shape: Tuple[int, ...], repeat_layout: TensorLayout):
    shape = tuple(x * y for x, y in zip(atom_shape, repeat_shape))
    layout = composition(TensorLayout(atom_shape, compact_col_major(shape)), make_layout(atom, complement(atom)))
    layout = logical_product(layout, make_layout(repeat_layout, complement(repeat_layout)))
    layout = make_layout(layout[0][0], layout[1][0])
    return shape, layout


def compact_coshape(shape: Tuple, layout: TensorLayout):
    """
    Compute the compact co-shape of a thread-value layout. The thread-value layout
    may be broadcasted in some dimensions, and the compact shape is the shape in which
    the broadcasted dimensions are removed. This operation is useful when we want to
    compute the shape of the codomain of a thread-value layout before broadcasting and
    adding extra zero-stride dimensions into this layout.

    Parameters:
       shape (Tuple): The shape of the tensor.
       layout (TensorLayout): The thread-value layout.

    Returns:
         Tuple: The compact shape of the thread-value layout.

    Example:
        >>> shape = (16, 8)
        >>> layout = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        >>> compact_coshape(shape, layout)
        (16, 8)
        The output shape is identical to the input shape since the layout is not
        broadcasted.
        >>> shape = (16, 8)
        >>> layout = TensorLayout(((4, 8), (2, 2)), ((32, 0), (16, 0)))
        >>> compact_coshape(shape, layout)
        (8, )
        The output shape is (8, ) since the layout is broadcasted in the first
        dimension.
    """
    flat_shape = flatten(layout.shape)
    flat_stride = flatten(layout.stride)
    sorted_ds = sorted(zip(flat_stride, flat_shape))
    col_major = compact_col_major(shape + (1,))[1:]
    result_shape = []
    result_stride = []
    idx = 0
    current_mode_s = []
    current_mode_d = []
    for d, s in sorted_ds:
        if d == 0:
            pass
        elif d * s < col_major[idx]:
            current_mode_s.append(s)
            current_mode_d.append(d)
        else:
            if d * s == col_major[idx]:
                current_mode_s.append(s)
                current_mode_d.append(d)
                result_shape.append(current_mode_s)
                result_stride.append(current_mode_d)
                current_mode_s = []
                current_mode_d = []
                idx = idx + 1
            else:
                s1 = shape_div(col_major[idx], d)
                current_mode_s.append(s1)
                current_mode_d.append(d)
                result_shape.append(current_mode_s)
                result_stride.append(current_mode_d)
                s2 = shape_div(s, s1)
                current_mode_s = [s2]
                current_mode_d = [d * s1]
                idx = idx + 1
    if len(current_mode_s) > 0:
        result_shape.append(current_mode_s)
        result_stride.append(current_mode_d)
    grouped = [TensorLayout(tuple(s), tuple(d)) for s, d in zip(result_shape, result_stride)]
    shape = tuple(filter(a).size() for a in grouped)
    return shape


def chain(
    atom_shape: Tuple[int, ...],
    atom_thrval_layout: TensorLayout,
    atom_repeat_shape: Tuple[int, ...],
    atom_repeat_layout: TensorLayout,
    levels: List[Level],
):
    shape, layout = zoom(atom_shape, atom_thrval_layout, atom_repeat_shape, atom_repeat_layout)
    current = None
    for level in levels:
        if current is None:
            current = level.level
        else:
            assert current == level.unit
            current = level.level
        shape, layout = zoom(shape, layout, level.shape, level.layout)
        layout = make_layout(
            make_layout(coalesce(make_layout(layout[0][0][0], layout[1])), layout[0][0][1]), layout[0][1]
        )
        shape, layout = zoom(shape, layout, level.repeat_shape, level.repeat_layout)
        layout = make_layout(layout[0][0], coalesce(make_layout(layout[0][1], layout[1])))
    return shape, layout


class TiledTensorLayout(LayoutBase):
    def __init__(self, atom: ThrValAtom, levels: List[Level] = None):
        if atom is None:
            assert levels is None
            self.atom = atom
            self.levels = []
            self.tensor_shape = None
            self.tv_layout = None
        else:
            self.atom = atom
            if levels is None:
                levels = []
            self.levels = sorted(levels, key=lambda x: x.level.value)
            self.tensor_shape, self.tv_layout = chain(
                self.atom.shape, self.atom.layout, self.atom.repeat_shape, self.atom.repeat_layout, self.levels
            )

    def str_indented(self, depth: int = 0):
        indent = " " * (depth * 2)
        prev_indent = " " * (max(0, depth - 1) * 2)
        return (
            "{"
            + f"\n{indent}atom: {self.atom.str_indented(depth+1)}, \n{indent}levels:["
            + ", ".join([f"{level.str_indented(depth+1)}" for level in self.levels])
            + f"]\n{prev_indent}"
            + "}"
        )

    def shape(self):
        return self.tensor_shape

    def thrval_layout(self):
        return self.tv_layout

    def thr_layout(self):
        return self.tv_layout[0][0]

    def val_layout(self):
        return coalesce(make_layout(self.tv_layout[0][1], self.tv_layout[1]))

    def val_count(self):
        return self.val_layout().count()

    def __eq__(self, other):
        left_thr_layout = coalesce(self.thr_layout())
        right_thr_layout = coalesce(other.thr_layout())
        return left_thr_layout == right_thr_layout and self.val_layout() == other.val_layout()


def canonicalize(a: TensorLayout):
    stride = canonicalize_uni_shape(a.shape, a.stride)
    return TensorLayout(a.shape, stride)


def codomain_from_shape_and_tv_layout(shape: Tuple[int, ...], tv_layout: TensorLayout):
    """
    Compute the codomain layout for a given tensor shape and thread-value layout.
    The mathematical definition of a codomain is the set of possible outputs of a function.
    The codomain is the set into which all outputs of the function are constrained to fall.
    In the context of a thread-value layout, the codomain layout refers to the layout of
    the multi-dimensional tile. The distribution of data within this tile across threads
    is described by the thread-value layout.

    Parameters:
        shape (Tuple[int, ...]): The shape of the tensor.
        tv_layout (TensorLayout): The thread-value layout.

    Returns:
        TensorLayout: The computed codomain layout.

    Example:
        >>> shape = (16, 8)
        >>> tv_layout = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        >>> codomain = codomain_from_shape_and_tv_layout(shape, tv_layout)
        >>> print(codomain)
        (16, 8):(1, 16)
        # The computed codomain layout is exactly the same as the column-major layout
        # generated from the shape (16, 8) because the thread-value layout is a surjective
        # function and there is no broadcasting in any dimension of the tensor.

        >>> shape = (16, 8)
        >>> tv_layout = TensorLayout(((4, 8), (2, 2)), ((32, 0), (16, 0)))
        >>> codomain = codomain_from_shape_and_tv_layout(shape, tv_layout)
        >>> print(codomain)
        (16, 8):(0, 16)
        # The second thread-value layout broadcasts in the first dimension of the tensor,
        # which can be derived from the thread-value layout. We extract this information
        # from the thread-value layout and reorganize the shapes and strides to align
        # with the tensor shape. Therefore, the result is reasonable and human-readable.

    """
    flat_shape = flatten(tv_layout.shape)
    flat_stride = flatten(tv_layout.stride)
    sorted_ds = sorted([(d, s) for d, s in zip(flat_stride, flat_shape) if d != 0])
    result_shape = []
    result_stride = []
    current_idx = 1
    total_idx = product(shape)
    if len(sorted_ds) > 0:
        nz_stride, nz_shape = list(zip(*sorted_ds))
        result_shape = []
        result_stride = []
        for s, d in zip(nz_shape, nz_stride):
            if d != current_idx:
                s1 = shape_div(d, current_idx)
                result_shape.append(s1)
                result_stride.append(0)
                current_idx *= s1
            result_shape.append(s)
            result_stride.append(d)
            current_idx *= s
    if current_idx != total_idx:
        s1 = shape_div(total_idx, current_idx)
        result_shape.append(s1)
        result_stride.append(0)
    codomain = coalesce(TensorLayout(tuple(result_shape), tuple(result_stride)))
    layouts = []
    for extent in shape[:-1]:
        layout, rest = group(codomain, extent)
        layouts.append(layout)
        codomain = rest
    layouts.append(codomain)
    return make_layout(*layouts)


def canonical_thread_value_layout(layout: TensorLayout):
    """
    Canonicalize the thread-value layout. This function separates the thread layout and
    creates the separated flattened thread-mode and value-mode of the input layouts
    """
    return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))
