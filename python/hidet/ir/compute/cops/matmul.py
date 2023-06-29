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
from typing import Sequence
from hidet.ir.expr import Expr, Var, is_constant
from hidet.ir.compute.primitives import TensorNode, compute, reduce
from hidet.ir.utils import broadcast_shape, broadcast_indices


def is_true(cond: Expr) -> bool:
    if is_constant(cond):
        return bool(cond) is True
    return False


def is_false(cond: Expr) -> bool:
    if is_constant(cond):
        return bool(cond) is False
    return False


def matmul(a: TensorNode, b: TensorNode, allow_1d=False) -> TensorNode:
    # The semantics of this operator is the same as the one in numpy
    # See Also https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    if len(a.shape) <= 1 and len(b.shape) <= 1:
        raise ValueError('At least one of the inputs must have rank > 1')
    if (len(a.shape) < 2 or len(b.shape) < 2) and not allow_1d:
        raise ValueError('Both inputs must have rank >= 2')
    elif len(a.shape) == 1:
        if is_true(a.shape[0] != b.shape[-2]):
            raise ValueError('Cannot multiply matrices with shape {} and {}.'.format(a.shape, b.shape))
        reduce_extent = a.shape[0]
        c_shape = b.shape[:-2] + b.shape[-1:]
    elif len(b.shape) == 1:
        if is_true(a.shape[-1] != b.shape[0]):
            raise ValueError('Cannot multiply matrices with shape {} and {}.'.format(a.shape, b.shape))
        reduce_extent = a.shape[-1]
        c_shape = a.shape[:-1]
    else:
        if is_true(a.shape[-1] != b.shape[-2]):
            raise ValueError('Cannot multiply matrices with shape {} and {}.'.format(a.shape, b.shape))
        reduce_extent = a.shape[-1]
        c_shape = broadcast_shape(a.shape[:-2], b.shape[:-2]) + [a.shape[-2], b.shape[-1]]

    def fcompute(indices: Sequence[Var], k: Var) -> Expr:
        indices = list(indices)
        if len(a.shape) == 1:
            a_val = a[k]
            b_val = b[indices[:-1] + [k] + indices[-1:]]
        elif len(b.shape) == 1:
            a_val = a[indices + [k]]
            b_val = b[k]
        else:
            a_indices = broadcast_indices(indices[:-2], a.shape[:-2], c_shape[:-2])
            b_indices = broadcast_indices(indices[:-2], b.shape[:-2], c_shape[:-2])
            a_val = a[a_indices + [indices[-2], k]]
            b_val = b[b_indices + [k, indices[-1]]]
        return a_val * b_val

    c = compute(
        name='c',
        shape=c_shape,
        fcompute=lambda *indices: reduce(
            shape=[reduce_extent], fcompute=lambda k: fcompute(indices, k), reduce_type='sum'
        ),
    )
    return c
