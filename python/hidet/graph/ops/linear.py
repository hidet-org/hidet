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
from typing import List, Sequence
from hidet.ir import primitives
from hidet.ir import Var, expr, dtypes
from hidet.ir.type import DataType
from hidet.ir.expr import Expr, if_then_else, logical_or, is_constant, is_true
from hidet.ir.tools import rewrite
from hidet.utils import prod, same_list
from .utils import Task, Operator, Tensor, TensorNode, InverseMap, compute, input_like
from .utils import broadcast_shape, broadcast_shapes, broadcast_indices
from .utils import normalize_slice, normalize_dim

def einsum(equation: str, operands: Sequence[Tensor]):
    print(equation)
    print(operands)
    pass