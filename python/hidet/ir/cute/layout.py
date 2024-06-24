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
# Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
##################################################################################################/
# This file is a python implementation for the layout (core concept) in CuTe, which will be
# used for integrating CuTe dialect.
from typing import List, Tuple, Union, Optional, Callable
import enum
from enum import auto as enum_auto

from hidet.ir.expr import Expr, is_constant
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
)


# cute layout
# TODO: choose a better name
class TensorLayout:
    def __init__(self, shape, stride=None):
        self.shape = shape
        if stride is None:
            stride = prefix_product(shape)
        assert congruent(shape, stride)
        self.stride = stride

    @property
    def shape_tuple(self) -> Tuple[Union[int, Expr]]:
        if is_integer(self.shape):
            return tuple([self.shape])
        else:
            return self.shape

    @property
    def stride_tuple(self) -> Tuple[Union[int, Expr]]:
        if is_integer(self.stride):
            return tuple([self.stride])
        else:
            return self.stride

    def __str__(self):
        return f"{self.shape}:{self.stride}"

    def __getitem__(self, i):
        return TensorLayout(self.shape[i], self.stride[i])

    def __call__(self, i, base: Optional[Union[int, Expr]] = None) -> Expr:
        if isinstance(i, list):
            crd = tuple(i)
        elif is_integer(i):
            crd = idx2crd(i, self.shape)
        else:
            assert isinstance(i, tuple)
            crd = i
        assert (is_integer(crd) and is_integer(self.shape)) or len(crd) == len(self.shape)
        idx = crd2idx(crd, self.shape, self.stride)
        return base + idx if base is not None else idx

    def __eq__(self, other):
        left_shape = self.shape
        left_stride = self.stride
        right_shape = other.shape
        right_stride = other.stride
        if (not congruent(left_shape, right_shape)) or (not congruent(left_stride, right_stride)):
            return False
        left_shape = flatten(left_shape)
        left_stride = flatten(left_stride)
        right_shape = flatten(right_shape)
        right_stride = flatten(right_stride)
        if isinstance(left_stride, tuple):
            return all(
                s1 == s2 and d1 == d2 for s1, s2, d1, d2 in zip(left_shape, right_shape, left_stride, right_stride)
            )
        else:
            return left_shape == right_shape and left_stride == right_stride

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


class ComposedTensorLayout:
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
    def shape_tuple(self) -> Tuple[Union[int, Expr]]:
        return self.layout.shape_tuple

    @property
    def stride_tuple(self) -> Tuple[Union[int, Expr]]:
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
        b = self.base + base if base is not None else self.base
        return self.functor(b + self.layout(i))

    def __eq__(self, other):
        raise NotImplementedError

    def size(self):
        return self.layout.size()

    def cosize(self):
        return self.layout.cosize()

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
    return right_inverse(make_layout(a, complement(a)))


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

    while len(shape_a) > 0:
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

    def str_indented(self, depth: int = 0):
        raise NotImplementedError()


class CopyAtom(Atom):
    def __init__(
        self,
        level: Union[str, Label],
        shape: Tuple[int],
        src_thrval_layout: TensorLayout,
        dst_thrval_layout: TensorLayout = None,
        repeat_shape: Tuple[int] = None,
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


def chain(
    atom_shape: Tuple[int, ...],
    atom_thrval_layout: TensorLayout,
    atom_repeat_shape: Tuple[int, ...],
    atom_repeat_layout: TensorLayout,
    levels: List[Level],
):
    shape, layout = zoom(atom_shape, atom_thrval_layout, atom_repeat_shape, atom_repeat_layout)
    for level in levels:
        shape, layout = zoom(shape, layout, level.shape, level.layout)
        layout = make_layout(
            make_layout(coalesce(make_layout(layout[0][0][0], layout[1])), layout[0][0][1]), layout[0][1]
        )
        shape, layout = zoom(shape, layout, level.repeat_shape, level.repeat_layout)
        layout = make_layout(layout[0][0], coalesce(make_layout(layout[0][1], layout[1])))
    return shape, layout


class TiledTensorLayout:
    def __init__(self, atom: ThrValAtom, levels: List[Level] = None):
        self.atom = atom
        if levels is None:
            levels = []
        self.levels = levels
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
