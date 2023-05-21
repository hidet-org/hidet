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
from typing import List, Union, Tuple
from hidet.ir.dtypes import float32, int32
from hidet.ir.func import IRModule, Function
from hidet.ir.compute import TensorNode
from hidet.ir.stmt import DeclareScope
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.definitions.utils import input_like, broadcast_shape, can_mutually_broadcast
from hidet.graph.ops.definitions.utils import tune
from hidet.graph.operator import Operator, Tensor
from hidet.graph.ops.definitions.utils import broadcast_indices
from hidet.graph.ops.definitions.arithmetic import sqrt


class MatmulF32Taskx86V2(Task):
    def __init__(self, a: TensorNode, b: TensorNode):
        a_shape = a.const_shape
        b_shape = b.const_shape

        if not a.type.dtype == float32 or not b.type.dtype == float32:
            raise ValueError('Both inputs must be float32 tensors')

        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError('Matrix multiplication expect at least 2D tensor, got {} and {}'.format(a_shape, b_shape))

        if a_shape[-1] != b_shape[-2]:
            raise ValueError(
                'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                ', got {} and {}'.format(a_shape, b_shape)
            )
        k_size = a_shape[-1]
        c_shape = broadcast_shape(a_shape[:-2], b_shape[:-2]) + [a_shape[-2], b_shape[-1]]

        c = compute(
            name='c',
            shape=c_shape,
            fcompute=lambda *indices: reduce(
                shape=[k_size],
                fcompute=lambda k: a[broadcast_indices(indices[:-2], a_shape[:-2], c_shape[1:-2]) + [indices[-2], k]]
                                   * b[broadcast_indices(indices[:-2], b_shape[:-2], c_shape[1:-2]) + [k, indices[-1]]],
                reduce_type='sum'
            )
        )

        super().__init__(
            name='matmul_f32_x86', inputs=[a, b], outputs=[c], attributes={
                'm_size': a_shape[-2],
                'n_size': b_shape[-1],
                'k_size': a_shape[-1]
            }
        )


    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_matmulf32_x86)


    def schedule_matmulf32_x86(self, block_m=4032, block_n=96, block_k=96, nthrs=32, micro_ker=(6, 16)):
        import hidet
        from hidet.ir.type import tensor_type
        from hidet.lang import tensor, grid, as_tensor_pointer
        from hidet.lang.layout import row_layout, col_layout
        from hidet.lang.avx import avx_f32x8_store, avx_f32x8_fmadd, avx_f32x8_load, avx_f32x8_broadcast

        node_a, node_b, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape: Tuple[int] = node_a.const_shape
        b_shape: Tuple[int] = node_b.const_shape
        c_shape: Tuple[int] = node_c.const_shape
        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]

        tile_m, tile_n = micro_ker
        tune.check(block_m % tile_m == block_n % tile_n == 0, 'Tile size must divide the corresponding block size')

        # TODO: Do I still want to pack it? If so add variables here

        with hidet.script_module() as module:
            @hidet.script
            def calc_nthr_nocopy_avx(m: int, n: int, k: int):

                # returns: (nthrs_m, nthrs_n, nthrs_k, BM, BN, BK
                if nthrs == 1:
                    return 1, 1, 1, 1, 1, 1

                BM_NOCOPY_AVX = 64
                BN_NOCOPY_AVX = 48
                BK_NOCOPY_AVX = 384
                BN_LARGE_NOCOPY_AVX = 192
                BM_SMALL_NOCOPY_AVX = 16
                BN_SMALL_NOCOPY_AVX = 1
                BK_SMALL_NOCOPY_AVX = 4

                nthr = nthrs
                nthr_m = (m + BM_NOCOPY_AVX - 1) // BM_NOCOPY_AVX
                nthr_n = (n + BN_NOCOPY_AVX - 1) // BN_NOCOPY_AVX
                nthr_k = 1
                # Partition along K dimension if that's beneficial
                nthr_other = nthr_k = 1
                while nthr_m * nthr_n * nthr_other < nthr and \
                    k // (nthr_other + 1) > BK_NOCOPY_AVX:
                    nthr_other += 1
                    if (nthr // nthr_other) * nthr_other > 0.9 * nthr:
                        nthr_k = nthr_other

                nthr = nthr // nthr_k
                if nthr_m == 1:
                    nthr_n = nthr
                if nthr_n == 1:
                    nthr_m = nthr

                # Simple partition reduction
                while nthr_m * nthr_n > nthr:
                    if nthr_m > nthr_n:
                        nthr_m -= 1
                    else:
                        nthr_n -= 1
                while nthr_m * nthr_n < nthr:
                    if nthr_m < nthr_n:
                        nthr_m += 1
                    else:
                        nthr_n += 1

                if nthr_m * nthr_n > nthr and nthr_m > 1 and nthr_n > 1:
                    if nthr_m <= nthr_n:
                        nthr_m = int32(sqrt(float32(nthr)))
                        if nthr_m > (m + BM_SMALL_NOCOPY_AVX - 1) // BM_SMALL_NOCOPY_AVX:
                            nthr_m = (m + BM_SMALL_NOCOPY_AVX - 1) // BM_SMALL_NOCOPY_AVX
                        nthr_n = nthr // nthr_m

                        while nthr_m > 1 and nthr_m * nthr_n != nthr:
                            nthr_m -= 1
                            nthr_n = nthr // nthr_m
                    else:
                        nthr_n = int32(sqrt(float32(nthr)))
                        if nthr_n > (n + BN_SMALL_NOCOPY_AVX - 1) // BN_SMALL_NOCOPY_AVX:
                            nthr_n = (n + BN_SMALL_NOCOPY_AVX - 1) // BN_SMALL_NOCOPY_AVX
                        nthr_m = nthr // nthr_n

                        while nthr_n > 1 and nthr_m * nthr_n != nthr:
                            nthr_n -= 1
                            nthr_m = nthr // nthr_n

                # TODO: Finish the resting starting with MB = ... tomorrow!











































