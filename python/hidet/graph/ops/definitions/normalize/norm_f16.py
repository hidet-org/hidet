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
from typing import List, Union, Optional, Sequence
from hidet.ir import IRModule, dtypes
from hidet.ir.primitives import active_mask, shfl_down_sync, shfl_sync
from hidet.ir.compute import ReduceOperation, reduce
from hidet.ir.type import data_type
from hidet.ir.layout import DataLayout
from hidet.lang import f16, f32, spatial, repeat, attrs, tensor_pointer
from hidet.lang.cuda import blockIdx, threadIdx, register_tensor
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, ReduceType, normalize_dim
from hidet.graph.ops.definitions.utils import compute, input_like, normalize_dim
from hidet.graph.ops.definitions.arithmetic import square, rsqrt
from hidet.utils import prod
from hidet.ir import primitives as prim
from .norm import NormalizeTask


class NormalizeF16Task(NormalizeTask):
    """
    Performs the following operation in float16 precision
        mean = x.mean(dims, keep_dim=True)
        x = x - mean
        variance = square(x).mean(dims, keep_dim=True)
        x = x * rsqrt(variance + epsilon)
    """

    def implement_cuda(self, working_dir: str) -> IRModule:
        return NotImplemented  # todo


class NormalizeF16Op(Operator):
    def __init__(self, x: Tensor, dims, epsilon: float, accumulate_dtype: str):
        rank = len(x.shape)
        dims = normalize_dim(dims, rank=rank)
        super().__init__(
            inputs=[x],
            attributes={'dims': dims, 'epsilon': epsilon, 'accumulate_dtype': accumulate_dtype},
            task=NormalizeF16Task(input_like(x, 'x'), dims, epsilon, accumulate_dtype),
        )


def normalize_f16(x: Tensor, axis: List[int], epsilon: float = 1e-5, accumulate_dtype: str = 'float32') -> Tensor:
    """Instance norm.

    Parameters
    ----------
    x: Tensor
        The data to be normalized.
    axis: int
        The axis of channel dimension.
    epsilon: float
        The epsilon added to variance.
    accumulate_dtype: str
        The precision used for accumulation during reduction

    Returns
    -------
    ret: Tensor
        The normalized tensor.
    """
    return NormalizeF16Op(x, axis, epsilon, accumulate_dtype).get_output(0)
