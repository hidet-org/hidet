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
from hidet.ir import Expr
from .int_tuple import (
    signum,
    depth,
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
)


def is_integer(i):
    return isinstance(i, (int, Expr))


# cute layout
# TODO: choose a better name
class TensorLayout:
    def __init__(self, shape, stride=None):
        self.shape = shape
        if stride is None:
            stride = prefix_product(shape)
        assert congruent(shape, stride)
        self.stride = stride

    def __str__(self):
        return f"{self.shape}:{self.stride}"

    def __getitem__(self, i):
        return TensorLayout(self.shape[i], self.stride[i])

    def __call__(self, i) -> Expr:
        if isinstance(i, list):
            crd = tuple(i)
        elif is_integer(i):
            crd = idx2crd(i, self.shape)
        else:
            assert isinstance(i, tuple)
            crd = i
        assert (is_integer(i) and is_integer(self.shape)) or len(crd) == len(self.shape)
        return crd2idx(crd, self.shape, self.stride)

    def size(self):
        return size(self.shape)

    def cosize(self):
        flat_shape = flatten(self.shape)
        flat_stride = flatten(self.stride)
        if is_integer(flat_stride):
            return (flat_shape - 1) * abs(flat_stride) + 1
        else:
            abs_sub_layout = TensorLayout(flat_shape, tuple(abs(i) for i in flat_stride))
            return abs_sub_layout(abs_sub_layout.size() - 1) + 1

    def depth(self):
        return depth(self.shape)

    def reversed(self):
        return TensorLayout(tuple(reversed(self.shape)), tuple(reversed(self.stride)))


def make_layout(*layouts):
    return TensorLayout(tuple(layout.shape for layout in layouts), tuple(layout.stride for layout in layouts))


def coalesce(a: TensorLayout):
    if is_integer(a.shape):
        return a if a.shape != 1 else TensorLayout(1, 0)
    else:
        flat_shape = flatten(a.shape)
        flat_stride = flatten(a.stride)
        result_shape = []
        result_stride = []
        for s, d in zip(flat_shape, flat_stride):
            if len(result_shape) == 0:
                if s != 1:
                    result_shape.append(s)
                    result_stride.append(d)
            else:
                curr_shape = result_shape[-1]
                curr_stride = result_stride[-1]
                if d == curr_shape * curr_stride:
                    result_shape[-1] = curr_shape * s
                elif s != 1:
                    result_shape.append(s)
                    result_stride.append(d)
        if len(result_shape) == 0:
            return TensorLayout(1)
        elif len(result_shape) == 1:
            return TensorLayout(result_shape[0], result_stride[0])
        else:
            return TensorLayout(tuple(result_shape), tuple(result_stride))


def filter(a: TensorLayout):
    return coalesce(TensorLayout(filter_zeros(a.stride, a.shape), a.stride))


def composition(a: TensorLayout, b: TensorLayout):
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
                result_shape.append(min(abs(s), rest_shape))
                rest_shape = shape_div(rest_shape, abs(s))
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
                s2 = min(abs(s1), rest_shape)
                rest_shape = shape_div(rest_shape, abs(s1))
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
    if is_integer(filter_stride) and filter_stride == 0:
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
        flat_stride = [abs(i) for i in flat_layout.stride]
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
