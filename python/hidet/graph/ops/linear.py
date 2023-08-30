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
from .utils import Tensor
from .matmul import matmul

# ToDo: Actually fully implement einsum, supporting same usage as Numpy and Torch

# Do ad-hoc pattern matching: only support simple cases such as matrix multiply
def einsum(equation: str, operands: Sequence[Tensor]):
    if '...' in equation:
        raise NotImplementedError('einsum currently does not support ellipsis')
    if len(operands) != 2:
        raise NotImplementedError('einsum currently only supports 2 operands')

    a = operands[0]
    b = operands[1]
    equation = equation.replace(' ', '')
    lhs, rhs = equation.split('->')
    a_subs, b_subs = lhs.split(',')

    if len(rhs) != len(a_subs) or len(a_subs) != len(b_subs):
        raise NotImplementedError('einsum currently only supports inputs and output of same rank')

    a_batch, a_dims = a_subs[:-2], a_subs[-2:]
    b_batch, b_dims = b_subs[:-2], b_subs[-2:]
    c_batch, c_dims = rhs[:-2], rhs[-2:]

    if a_batch != b_batch or a_batch != c_batch:
        raise NotImplementedError('einsum currently only supports batched matmul')

    if a_dims[1] == b_dims[0]:
        c = matmul(a, b)
    elif a_dims[1] == b_dims[1]:
        c = matmul(a, b.transpose(-1, -2))
    elif a_dims[0] == b_dims[0]:
        c = matmul(a.transpose(-1, -2), b)
    elif a_dims[0] == b_dims[1]:
        c = matmul(a.transpose(-1, -2), b.transpose(-1, -2))
    else:
        raise NotImplementedError('einsum currently only supports batched matmul')

    transpose_c = (c_dims[0] == b_dims[0] or c_dims[0] == b_dims[1]) and (
        c_dims[1] == a_dims[0] or c_dims[1] == a_dims[1]
    )

    if transpose_c:
        return c.transpose(-1, -2)
    else:
        return c
