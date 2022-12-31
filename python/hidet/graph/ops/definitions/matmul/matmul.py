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
from typing import List, Tuple
from hidet.ir.expr import Expr, Var
from ..utils import Task, Operator, Tensor, TensorNode, compute, reduce, input_like, broadcast_shape, broadcast_indices


class MatmulTask(Task):
    def __init__(self, a: TensorNode, b: TensorNode):
        # The semantics of this operator is the same as the one in numpy
        # See Also https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        a_shape: List[int] = a.const_shape()
        b_shape: List[int] = b.const_shape()
        if len(a_shape) <= 1 and len(b_shape) <= 1:
            raise ValueError('At least one of the inputs must have rank > 1')
        elif len(a_shape) == 1:
            if a_shape[0] != b_shape[-2]:
                raise ValueError('Cannot multiply matrices with shape {} and {}.'.format(a_shape, b_shape))
            reduce_extent = a_shape[0]
            c_shape = b_shape[:-2] + b_shape[-1:]
        elif len(b_shape) == 1:
            if a_shape[-1] != b_shape[0]:
                raise ValueError('Cannot multiply matrices with shape {} and {}.'.format(a_shape, b_shape))
            reduce_extent = a_shape[-1]
            c_shape = a_shape[:-1]
        else:
            if a_shape[-1] != b_shape[-2]:
                raise ValueError('Cannot multiply matrices with shape {} and {}.'.format(a_shape, b_shape))
            reduce_extent = a_shape[-1]
            c_shape = broadcast_shape(a_shape[:-2], b_shape[:-2]) + [a_shape[-2], b_shape[-1]]

        def fcompute(indices: Tuple[Var], k: Var) -> Expr:
            if len(a_shape) == 1:
                a_val = a[k]
                b_val = b[indices[:-1] + (k,) + indices[-1:]]
            elif len(b_shape) == 1:
                a_val = a[indices + (k,)]
                b_val = b[k]
            else:
                a_indices = broadcast_indices(indices[:-2], a_shape[:-2], c_shape[:-2])
                b_indices = broadcast_indices(indices[:-2], b_shape[:-2], c_shape[:-2])
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
        super().__init__(name='matmul', inputs=[a, b], outputs=[c])


class MatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        task = MatmulTask(input_like(a, 'a'), input_like(b, 'b'))
        super().__init__(inputs=[a, b], task=task)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatmulOp(a, b).get_output(0)
