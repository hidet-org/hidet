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
from typing import List, Tuple, Union
from hidet.ir import dtypes
from hidet.ir.dtypes import float32
from hidet.ir.expr import if_then_else
from hidet.ir.func import IRModule, Function
from hidet.ir.compute import TensorNode
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.definitions.utils import input_like, broadcast_shape, can_mutually_broadcast
from hidet.graph.ops.definitions.utils import tune
from hidet.graph.operator import Operator, Tensor
from hidet.utils.py import is_power_of_two, cdiv, prod
from hidet.graph.ops.definitions.utils import broadcast_indices


class MatmulF32Taskx86(Task):
    def __init__(self, a: TensorNode, b: TensorNode):
        a_shape = a.const_shape()
        b_shape = b.const_shape()

        if not a.type.dtype == float32 or not b.type.dtype == float32:
            raise ValueError('Both inputs must be float32 tensors')

        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError('Matrix multiplication expect at least 2D tensor, got {} and {}'.format(a_shape, b_shape))
        if a_shape[-1] != b_shape[-2]:
            raise ValueError(
                'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                ', got {} and {}'.format(a_shape, b_shape)
            )
        if not can_mutually_broadcast(a_shape[:-2], b_shape[:-2]):
            raise ValueError(
                'Matrix multiplication expect tensor A and B with compatible broadcast shape, '
                'got {} and {}'.format(a_shape, b_shape)
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
                'n_size': b_shape[-2],
                'k_size': a_shape[-1]
            }
        )

    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return IRModule()  # TODO: Go back to it later

    def schedule(self, block_m=2014, block_n=512, block_k=768, micro_ker: str = '6x16', nthreads=8) -> IRModule:
        import hidet
        from hidet.ir.type import tensor_type
        from hidet.lang import attr, col_spatial, view, u32, tensor_pointer, grid, as_tensor_pointer
        from hidet.lang.layout import row_layout, col_layout
        from hidet.lang.mapping import spatial, auto_map
        from hidet.lang.avx import avx_f32x8_store, avx_f32x8_fmadd, avx_f32x8_load, avx_f32x8_broadcast
        from hidet.lang.avx import avx_f32x4_broadcast, avx_f32x4_fmadd, avx_f32x4_load

        node_a, node_b, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape: List[int] = node_a.const_shape()
        b_shape: List[int] = node_b.const_shape()
        c_shape: List[int] = node_c.const_shape()
        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]
        a_head, b_head, c_head = a_shape[:-2], b_shape[:-2], c_shape[:-2]

        supported_microkers = ('6x16', '4x8', '8x8')
        tune.check(micro_ker in supported_microkers, "The size of the micro-kernel is not supported")

        x_idx = micro_ker.find('x')
        tile_m = int(micro_ker[:x_idx])
        tile_n = int(micro_ker[x_idx+1:])
        tune.check(block_m % tile_m == block_n % tile_n == 0, 'Tile size must divide the corresponding block size')

        packed_a_type = tensor_type(
            'float32', layout=row_layout(block_m // tile_m, 1) * col_layout(tile_m, block_k)
        )
        packed_b_type = tensor_type(
            'float32', layout=row_layout(1, block_n // tile_n) * row_layout(block_k, tile_n)
        )
        c_type = tensor_type(
            'float32', shape=[m_size, n_size]
        )

        temp_c = tensor_type

        # TODO: Do I need any mappings? Since I think the coordination is automatically done
        # TODO: by openmp

        with hidet.script_module() as module:
            @hidet.script
            def micro_kernel_6x16(a: ~packed_a_type,
                                  b: ~packed_b_type,
                                  c: ~c_type,
                                  pb: int):
                c0 = avx_f32x8_load(~c[0, 0])
                c08 = avx_f32x8_load(~c[0, 8])
                c1 = avx_f32x8_load(~c[1, 0])
                c18 = avx_f32x8_load(~c[1, 8])
                c2 = avx_f32x8_load(~c[2, 0])
                c28 = avx_f32x8_load(~c[2, 8])
                c3 = avx_f32x8_load(~c[3, 0])
                c38 = avx_f32x8_load(~c[3, 8])
                c4 = avx_f32x8_load(~c[4, 0])
                c48 = avx_f32x8_load(~c[4, 8])
                c5 = avx_f32x8_load(~c[5, 0])
                c58 = avx_f32x8_load(~c[5, 8])

                for pp in range(pb):
                    bb0to7 = avx_f32x8_load(~b[pp, 0])
                    bb8to15 = avx_f32x8_load(~b[pp, 8])

                    aa = avx_f32x8_broadcast(~a[0, pp])
                    c0 = avx_f32x8_fmadd(aa, bb0to7, c0)
                    c08 = avx_f32x8_fmadd(aa, bb8to15, c08)
                    aa = avx_f32x8_broadcast(~a[1, pp])
                    c1 = avx_f32x8_fmadd(aa, bb0to7, c1)
                    c18 = avx_f32x8_fmadd(aa, bb8to15, c18)
                    aa = avx_f32x8_broadcast(~a[2, pp])
                    c2 = avx_f32x8_fmadd(aa, bb0to7, c2)
                    c28 = avx_f32x8_fmadd(aa, bb8to15, c28)
                    aa = avx_f32x8_broadcast(~a[3, pp])
                    c3 = avx_f32x8_fmadd(aa, bb0to7, c3)
                    c38 = avx_f32x8_fmadd(aa, bb8to15, c38)
                    aa = avx_f32x8_broadcast(~a[4, pp])
                    c4 = avx_f32x8_fmadd(aa, bb0to7, c4)
                    c48 = avx_f32x8_fmadd(aa, bb8to15, c48)
                    aa = avx_f32x8_broadcast(~a[5, pp])
                    c5 = avx_f32x8_fmadd(aa, bb0to7, c5)
                    c58 = avx_f32x8_fmadd(aa, bb8to15, c58)
                avx_f32x8_store(~c[0, 0], c0)
                avx_f32x8_store(~c[0, 8], c08)
                avx_f32x8_store(~c[1, 0], c1)
                avx_f32x8_store(~c[1, 8], c18)
                avx_f32x8_store(~c[2, 0], c2)
                avx_f32x8_store(~c[2, 8], c28)
                avx_f32x8_store(~c[3, 0], c3)
                avx_f32x8_store(~c[3, 8], c38)
                avx_f32x8_store(~c[4, 0], c4)
                avx_f32x8_store(~c[4, 8], c48)
                avx_f32x8_store(~c[5, 0], c5)
                avx_f32x8_store(~c[5, 8], c58)

            @hidet.script
            def micro_kernel_4x8(a: ~packed_a_type,
                                  b: ~packed_b_type,
                                  c: ~c_type,
                                  pb: int):
                c0 = avx_f32x8_load(~c[0, 0])
                c1 = avx_f32x8_load(~c[1, 0])
                c2 = avx_f32x8_load(~c[2, 0])
                c3 = avx_f32x8_load(~c[3, 0])

                for pp in range(pb):
                    bb = avx_f32x8_load(~b[pp, 0])

                    aa = avx_f32x8_broadcast(~a[0, pp])
                    c0 = avx_f32x8_fmadd(aa, bb, c0)
                    aa = avx_f32x8_broadcast(~a[1, pp])
                    c1 = avx_f32x8_fmadd(aa, bb, c1)
                    aa = avx_f32x8_broadcast(~a[2, pp])
                    c2 = avx_f32x8_fmadd(aa, bb, c2)
                    aa = avx_f32x8_broadcast(~a[3, pp])
                    c3 = avx_f32x8_fmadd(aa, bb, c3)
                avx_f32x8_store(~c[0, 0], c0)
                avx_f32x8_store(~c[1, 0], c1)
                avx_f32x8_store(~c[2, 0], c2)
                avx_f32x8_store(~c[3, 0], c3)

            @hidet.script
            def micro_kernel_8x8(a: ~packed_a_type,
                                 b: ~packed_b_type,
                                 c: ~c_type,
                                 pb: int):
                c0 = avx_f32x8_load(~c[0, 0])
                c1 = avx_f32x8_load(~c[1, 0])
                c2 = avx_f32x8_load(~c[2, 0])
                c3 = avx_f32x8_load(~c[3, 0])
                c4 = avx_f32x8_load(~c[4, 0])
                c5 = avx_f32x8_load(~c[5, 0])
                c6 = avx_f32x8_load(~c[6, 0])
                c7 = avx_f32x8_load(~c[7, 0])

                for pp in range(pb):
                    bb = avx_f32x8_load(~b[pp, 0])

                    aa = avx_f32x8_broadcast(~a[0, pp])
                    c0 = avx_f32x8_fmadd(aa, bb, c0)
                    aa = avx_f32x8_broadcast(~a[1, pp])
                    c1 = avx_f32x8_fmadd(aa, bb, c1)
                    aa = avx_f32x8_broadcast(~a[2, pp])
                    c2 = avx_f32x8_fmadd(aa, bb, c2)
                    aa = avx_f32x8_broadcast(~a[3, pp])
                    c3 = avx_f32x8_fmadd(aa, bb, c3)
                    aa = avx_f32x8_broadcast(~a[4, pp])
                    c4 = avx_f32x8_fmadd(aa, bb, c4)
                    aa = avx_f32x8_broadcast(~a[5, pp])
                    c5 = avx_f32x8_fmadd(aa, bb, c5)
                    aa = avx_f32x8_broadcast(~a[6, pp])
                    c6 = avx_f32x8_fmadd(aa, bb, c6)
                    aa = avx_f32x8_broadcast(~a[7, pp])
                    c7 = avx_f32x8_fmadd(aa, bb, c7)
                avx_f32x8_store(~c[0, 0], c0)
                avx_f32x8_store(~c[1, 0], c1)
                avx_f32x8_store(~c[2, 0], c2)
                avx_f32x8_store(~c[3, 0], c3)
                avx_f32x8_store(~c[4, 0], c4)
                avx_f32x8_store(~c[5, 0], c5)
                avx_f32x8_store(~c[6, 0], c6)
                avx_f32x8_store(~c[7, 0], c7)


            @hidet.script
            def macro_kernel(a: ~packed_a_type, b: ~packed_b_type, c: ~c_type,
                             ib: int, jb: int, pb: int):
                mpanels = (ib + tile_m - 1) // tile_m
                npanels = (ib + tile_n - 1) // tile_n
                _mr = ib % tile_m
                _nr = jb % tile_n

                microker_table = {
                    '6x16': micro_kernel_6x16,
                    '4x8': micro_kernel_4x8,
                    '8x8': micro_kernel_8x8
                }

                micro_kernel = microker_table[micro_ker]
                # Loop 2
                for mpanel in grid(mpanels, attrs=f'p{nthreads}'):
                    mr = tile_m if mpanel != mpanels - 1 or _mr == 0 else _mr
                    ii = mpanel * tile_m
                    # Loop 1
                    for npanel in range(npanels):
                        nr = tile_n if npanel != npanels - 1 or _nr == 0 else _nr
                        jj = npanel * tile_n
                        # micro-kernel
                        if mr == tile_m and nr == tile_n:
                            micro_kernel(~a[ii, 0], ~b[0, jj], ~c[ii, jj], pb)
                        else:






        return IRModule()
































