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
# %%
from hidet.ir import dtypes
from hidet.ir.dtypes import float16, int8
from hidet.ir.expr import Expr, cast
from hidet.ir.compute import TensorNode
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.utils import input_like, broadcast_shape, can_mutually_broadcast
from hidet.graph.operator import Operator, Tensor
from hidet.graph.ops.utils import broadcast_indices


class SymmetricQuantizedMatmulTask(Task):
    def __init__(self, a: TensorNode, weight: TensorNode, scale: TensorNode):

        self._assert(
            a.type.dtype == float16 and weight.type.dtype == int8, 'Expect a to be float16 and weight to be int8'
        )
        # weight.shape = [K, M], scale.shape = [M]
        # such that the quantization is done over K
        self._assert(scale.shape[0] == weight.shape[1])

        if len(a.shape) < 2 or len(weight.shape) != 2:
            raise ValueError('SymmetricQuantizedMatmul expect , got {} and {}'.format(a.shape, weight.shape))

        self._assert(
            a.shape[-1] == weight.shape[-2],
            msg=(
                'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                ', got {} and {}'.format(a.shape, weight.shape)
            ),
        )

        self._assert(
            can_mutually_broadcast(a.shape[:-2], weight.shape[:-2]),
            msg=(
                'Matrix multiplication expect tensor A and B with compatible broadcast shape, '
                'got {} and {}'.format(a.shape, weight.shape)
            ),
        )

        a_shape = a.shape
        b_shape = weight.shape
        k_size = a.shape[-1]
        c_shape = broadcast_shape(a.shape[:-2], weight.shape[:-2]) + [a_shape[-2], b_shape[-1]]
        c = compute(
            name='c',
            shape=c_shape,
            fcompute=lambda *indices: reduce(
                shape=[k_size],
                fcompute=lambda k: a[broadcast_indices(indices[:-2], a.shape[:-2], c_shape[:-2]) + [indices[-2], k]]
                * (
                    cast(
                        weight[broadcast_indices(indices[:-2], weight.shape[:-2], c_shape[:-2]) + [k, indices[-1]]],
                        float16,
                    )
                    * scale[indices[-1]]
                ),
                reduce_type='sum',
            ),
        )

        super().__init__(name='symmetric_quantized_matmul', inputs=[a, weight, scale], outputs=[c], attributes={})


class SymmetricQuantizedMatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor, scale: Tensor):
        task = SymmetricQuantizedMatmulTask(input_like(a, 'a'), input_like(b, 'b'), input_like(scale, 'scale'))
        super().__init__(inputs=[a, b, scale], attributes={}, task=task)


def symmetric_quant_matmul(a: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
    if len(a.shape) < 2 or len(weight.shape) < 2:
        raise ValueError('a and b must have at least 2 dimensions, got shape {} and {}'.format(a.shape, weight.shape))
    # TODO: impliment dynamic run-time shape assertion
    if not (isinstance(a.shape[-1], Expr) or isinstance(weight.shape[-1], Expr)) and (
        a.shape[-1] % 2 != 0 or weight.shape[-1] % 2 != 0
    ):
        raise ValueError('Expect the last dimension of the input tensors to be a multiple of 2')
    if a.dtype != dtypes.float16 or weight.dtype != dtypes.int8:
        raise ValueError('BatchMatmulF16Op only support float16, int8, got {} and {}'.format(a.dtype, weight.dtype))
    return SymmetricQuantizedMatmulOp(a, weight, scale).outputs[0]
