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
from ..utils import Tensor, normalize_dim
from ..arithmetic import rsqrt
from typing import List
from hidet.ir import primitives as prim
from hidet.ir.library import tune
from hidet.ir.module import IRModule
from hidet.ir.primitives import active_mask, shfl_down_sync
from hidet.ir.compute import reduce
from hidet.ir.expr import Expr, convert, is_constant, if_then_else
from hidet.ir.type import DataType
from hidet.ir.layout import row_major
from hidet.ir.dtypes.vector import VectorType
from hidet.lang import spatial, repeat, grid, cast, register_tensor
from hidet.lang import data_type, TensorType, tensor_pointer, address, i32, attrs
from hidet.lang.cuda import blockIdx, threadIdx
from hidet.lang.cuda import dynamic_shared_memory, syncthreads
from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.utils import compute, input_like, normalize_dim
from hidet.utils import prod

class LpNormTask(Task):
    """
    Performs the following operation
    """

    def __init__(self, x: TensorNode, p: float, dim: int, eps: float):
        pass

class LpNormOp(Operator):
    def __init__(self, x: Tensor, p: float, dim: int, eps: float):
        super().__init__(
            inputs=[x],
            attributes={'p': p, 'dim': dim, 'eps': eps},
            task=LpNormTask(input_like(x, 'x'), p, dim, eps),
        )

def lp_norm(x: Tensor, p=2.0, dim=1, eps=1e-12):
    """LP norm.

    Parameters
    ----------
    x: Tensor
        The data to be normalized.
    p: float
        The exponent value in the norm formulation.
    dim: int
        The dimension to reduce.
    eps: float
        Small value to avoid division by zero.

    Returns
    -------
    ret: Tensor
        The normalized tensor.
    """
    return LpNormOp(x, p, dim, eps).outputs[0]