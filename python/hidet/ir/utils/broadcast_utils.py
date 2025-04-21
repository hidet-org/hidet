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
from typing import Sequence, List

import hidet.option
from hidet.ir.expr import Expr, Int, is_constant
from hidet.utils import repeat_until_converge


def can_broadcast(src_shape: Sequence[Int], dst_shape: Sequence[Int]) -> bool:
    if len(dst_shape) < len(src_shape):
        return False
    src_shape = [1 for _ in range(len(dst_shape) - len(src_shape))] + list(src_shape)
    for a, b in zip(src_shape, dst_shape):
        if is_constant(a, b) and a not in [1, b]:
            return False
    return True


def can_mutually_broadcast(x_shape: Sequence[Int], y_shape: Sequence[Int]) -> bool:
    x_shape, y_shape = list(x_shape), list(y_shape)
    while len(x_shape) < len(y_shape):
        x_shape = [1] + x_shape
    while len(y_shape) < len(x_shape):
        y_shape = [1] + y_shape
    return all(p == q or p == 1 or q == 1 for p, q in zip(x_shape, y_shape) if is_constant(p, q))


def broadcast_shape(x_shape: Sequence[Int], y_shape: Sequence[Int]) -> List[Int]:
    """
    Broadcast two shapes with the same rule as numpy.
    Please refer to https://numpy.org/doc/stable/user/basics.broadcasting.html for details.
    """
    from hidet.ir.dtypes import int32

    x_shape, y_shape = list(x_shape), list(y_shape)
    orig_shapes = x_shape, y_shape
    while len(x_shape) < len(y_shape):
        x_shape = [int32(1)] + x_shape
    while len(y_shape) < len(x_shape):
        y_shape = [int32(1)] + y_shape
    result_shape = []
    for p, q in zip(x_shape, y_shape):
        # Case 1: one of the dimensions is the constant 1
        if is_constant(p) and p == 1:
            result_shape.append(q)
        elif is_constant(q) and q == 1:
            result_shape.append(p)
        # Case 2: both dimensions are constant
        elif is_constant(p, q):
            if p != q:
                raise ValueError(
                    'Cannot broadcast operands with shape {} and {}'.format(orig_shapes[0], orig_shapes[1])
                )
            result_shape.append(p)
        # Case 3: exactly one of the dimensions is constant, assume the symbolic dimension is 1
        elif is_constant(p):
            result_shape.append(p)
        elif is_constant(q):
            result_shape.append(q)
        # Case 4: both dimensions are symbolic, this is only allowed if the dimensions are the same expression or at
        # least one of them resolves to 1.
        else:
            if not hidet.option.get_option('debug_strict_broadcast_check'):
                # Assume p == q
                result_shape.append(p)
                continue

            from hidet.transforms.rule_based_simplifier import RuleBasedSimplifier

            simp = RuleBasedSimplifier()
            p = repeat_until_converge(simp, p)
            q = repeat_until_converge(simp, q)

            if is_constant(p) and p == 1:
                result_shape.append(q)
            elif is_constant(q) and q == 1:
                result_shape.append(p)
            else:
                diff = repeat_until_converge(simp, p - q)
                if not is_constant(diff) or diff != 0:
                    raise ValueError(
                        "Broadcasting between operands with symbolic shapes {} and {} is ambiguous,"
                        " consider explicitly broadcasting before the operator to resolve this ambiguity".format(
                            *orig_shapes
                        )
                    )
                result_shape.append(p)

    return result_shape


def broadcast_shapes(shapes: Sequence[Sequence[Int]]) -> List[Int]:
    assert len(shapes) >= 1
    expanded_shape = list(shapes[0])
    for shape in shapes:
        expanded_shape = broadcast_shape(expanded_shape, shape)
    return expanded_shape


def broadcast_indices(out_indices: Sequence[Int], shape: Sequence[Int], out_shape: Sequence[Int]) -> List[Expr]:
    if len(out_indices) != len(out_shape):
        raise ValueError('Number of indices {} does not match the output shape {}'.format(out_indices, out_shape))

    pad_dim = len(out_shape) - len(shape)
    indices = list(out_indices[pad_dim:])
    for idx, dim in enumerate(shape):
        if dim == 1:
            indices[idx] = 0
    return indices
