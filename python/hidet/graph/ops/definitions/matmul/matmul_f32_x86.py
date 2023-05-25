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
from hidet.ir.primitives.math import sqrt, pow


class MatmulF32Taskx86(Task):
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
                'n_size': b_shape[-1],
                'k_size': a_shape[-1]
            }
        )

    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_matmulf32_x86)

    # @tune.space(2, 'micro_ker', [(6, 16)])
    # @tune.space(2, 'block_m', [1200, 2400])
    # @tune.space(2, 'block_n', [96, 192, 384, 512])
    # @tune.space(2, 'block_k', [128, 256, 384, 512])
    # @tune.space(2, 'nthreads', [2, 4, 8, 16, 32])
    # @tune.space(2, 'block_m', [2016])
    @tune.space(2, 'block_n', [48, 72, 144, 196, 256, 288, 360])
    @tune.space(2, 'block_k', [64, 72, 96, 128, 256, 512])
    @tune.space(2, 'block_m', [2016, 3000])
    @tune.space(2, 'nthreads', [4, 8, 16, 32])
    def schedule_matmulf32_x86(self, block_m=2016, block_n=360, block_k=512, micro_ker=(6, 16),
                               nthreads=16) -> IRModule:
        import hidet
        from hidet.ir.type import tensor_type
        from hidet.lang import col_spatial, tensor, u32, tensor_pointer, grid, as_tensor_pointer
        from hidet.lang.layout import row_layout, col_layout
        from hidet.lang.avx import avx_f32x8_store, avx_f32x8_fmadd, avx_f32x8_load, avx_f32x8_broadcast
        from hidet.lang.avx import avx_f32x4_broadcast, avx_f32x4_fmadd, avx_f32x4_load, avx_f32x4_store

        node_a, node_b, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape = node_a.const_shape
        b_shape = node_b.const_shape
        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]

        tile_m, tile_n = micro_ker

        supported_microkers = ((6, 16), (4, 8), (8, 8))
        tune.check(micro_ker in supported_microkers, "The size of the micro-kernel is not supported")

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

        aip_outer_rows = block_m // tile_m
        bip_outer_cols = block_n // tile_n

        with hidet.script_module() as module:
            @hidet.script
            def micro_kernel_6x16(a: packed_a_type,
                                  b: packed_b_type,
                                  c_ptr: ~float32,
                                  pb: int32,
                                  msize: int32,
                                  nsize: int32):
                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[msize, nsize])
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
            def micro_kernel_4x8(a: packed_a_type,
                                 b: packed_b_type,
                                 c_ptr: ~float32,
                                 pb: int32,
                                 msize: int32,
                                 nsize: int32):

                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[msize, nsize])
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
            def micro_kernel_8x8(a: packed_a_type,
                                 b: packed_b_type,
                                 c_ptr: ~float32,
                                 pb: int32,
                                 msize: int32,
                                 nsize: int32):

                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[msize, nsize])
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
            def micro_kernel_4x4(a: packed_a_type,
                                 b: packed_b_type,
                                 c_ptr: ~float32,
                                 pb: int32,
                                 msize: int32,
                                 nsize: int32):
                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[msize, nsize])

                c0 = avx_f32x4_load(~c[0, 0])
                c1 = avx_f32x4_load(~c[1, 0])
                c2 = avx_f32x4_load(~c[2, 0])
                c3 = avx_f32x4_load(~c[3, 0])

                for pp in range(pb):
                    bb = avx_f32x4_load(~b[pp, 0])

                    aa = avx_f32x4_broadcast(~a[0, pp])
                    c0 = avx_f32x4_fmadd(aa, bb, c0)
                    aa = avx_f32x4_broadcast(~a[1, pp])
                    c1 = avx_f32x4_fmadd(aa, bb, c1)
                    aa = avx_f32x4_broadcast(~a[2, pp])
                    c2 = avx_f32x4_fmadd(aa, bb, c2)
                    aa = avx_f32x4_broadcast(~a[3, pp])
                    c3 = avx_f32x4_fmadd(aa, bb, c3)
                avx_f32x4_store(~c[0, 0], c0)
                avx_f32x4_store(~c[1, 0], c1)
                avx_f32x4_store(~c[2, 0], c2)
                avx_f32x4_store(~c[3, 0], c3)

            micro_kernel = micro_kernel_6x16
            if tile_m == 8 and tile_n == 8:
                micro_kernel = micro_kernel_8x8
            elif tile_m == 4 and tile_n == 8:
                micro_kernel = micro_kernel_4x8
            elif tile_m == 4 and tile_n == 4:
                micro_kernel = micro_kernel_4x4

            @hidet.script
            def macro_kernel(a: packed_a_type, b: packed_b_type, c_in_macro: c_type,
                             ib: int32, jb: int32, pb: int32):
                mpanels = (ib + tile_m - 1) // tile_m
                npanels = (jb + tile_n - 1) // tile_n
                _mr = ib % tile_m
                _nr = jb % tile_n

                # Loop 2
                para = 'p' + str(nthreads)
                for mpanel in grid(mpanels, attrs=para):
                    mr = tile_m if mpanel != mpanels - 1 or _mr == 0 else _mr
                    ii = mpanel * tile_m
                    # Loop 1
                    for npanel in range(npanels):
                        nr = tile_n if npanel != npanels - 1 or _nr == 0 else _nr
                        jj = npanel * tile_n
                        # micro-kernel
                        if mr == tile_m and nr == tile_n:
                            micro_kernel(~a[ii, 0], ~b[0, jj], ~c_in_macro[ii, jj], pb, m_size, n_size)
                        else:
                            temp_c = tensor(
                                scope=DeclareScope.Default,
                                dtype='float32',
                                layout=row_layout(tile_m, tile_n)
                            )
                            for tempi in range(tile_m):
                                for tempj in range(tile_n):
                                    temp_c[tempi, tempj] = 0.0
                            micro_kernel(~a[ii, 0], ~b[0, jj], temp_c, pb, tile_m, tile_n)
                            for remain_row, remain_col in grid(mr, nr):
                                c_in_macro[ii + remain_row, jj + remain_col] += temp_c[remain_row, remain_col]

            @hidet.script
            def pack_a(a_ptr: ~float32, packed_a: packed_a_type, ib: int32, pb: int32):
                a = as_tensor_pointer(a_ptr, dtype=float32,
                                      shape=[m_size, k_size])

                mp = ib // tile_m
                mr = ib % tile_m
                for micropanel_idx in range(mp):
                    panel_row_start = micropanel_idx * tile_m
                    for micropanel_col in range(pb):
                        for micropanel_row in range(tile_m):
                            packed_a[micropanel_row + panel_row_start, micropanel_col] = \
                                a[micropanel_row + panel_row_start, micropanel_col]
                # pack the remaining if the shape is not nice
                if mr > 0:
                    remain_start_row = mp * tile_m
                    for remain_col in range(pb):
                        for remain_row in range(mr):
                            packed_a[remain_start_row + remain_row, remain_col] = \
                                a[remain_start_row + remain_row, remain_col]
                        remain_row = mr
                        while remain_row < tile_m:
                            packed_a[remain_start_row + remain_row, remain_col] = 0.0
                            remain_row += 1

            @hidet.script
            def pack_b(b_ptr: ~float32, packed_b: packed_b_type, jb: int32, pb: int32):
                np = jb // tile_n
                nr = jb % tile_n
                b = as_tensor_pointer(b_ptr, dtype=float32, shape=[k_size, n_size])
                for micropanel_idx in range(np):
                    panel_col_start = micropanel_idx * tile_n
                    for micropanel_row in range(pb):
                        for micropanel_col in range(tile_n):
                            packed_b[micropanel_row, micropanel_col + panel_col_start] = \
                                b[micropanel_row, micropanel_col + panel_col_start]
                if nr > 0:
                    remain_col_start = np * tile_n
                    for remain_row in range(pb):
                        for remain_col in range(nr):
                            packed_b[remain_row, remain_col + remain_col_start] = \
                                b[remain_row, remain_col + remain_col_start]
                        remain_col = nr
                        while remain_col < tile_n:
                            packed_b[remain_row, remain_col + remain_col_start] = 0.0
                            remain_col += 1

            @hidet.script
            def matmul_kernel_x86(
                    a_ptr: ~float32, b_ptr: ~float32, c_ptr: ~float32
            ):
                a = as_tensor_pointer(a_ptr, dtype=float32, shape=[m_size, k_size])
                b = as_tensor_pointer(b_ptr, dtype=float32, shape=[k_size, n_size])
                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[m_size, n_size])
                mbs = (m_size + block_m - 1) // block_m
                nbs = (n_size + block_n - 1) // block_n
                kbs = (k_size + block_k - 1) // block_k

                packed_a = tensor(
                    scope=DeclareScope.Default,
                    dtype=float32,
                    layout=row_layout(aip_outer_rows, 1) * col_layout(tile_m, block_k)
                )

                packed_b = tensor(
                    scope=DeclareScope.Default,
                    dtype=float32,
                    layout=row_layout(1, bip_outer_cols) * row_layout(block_k, tile_n)
                )

                for mb in range(mbs):
                    i = mb * block_m
                    ib = min(block_m, m_size - i)
                    for kb in range(kbs):
                        p = kb * block_k
                        pb = min(block_k, k_size - p)

                        mp = ib // tile_m
                        mr = ib % tile_m
                        for micropanel_idx in range(mp):
                            panel_row_start = micropanel_idx * tile_m
                            for micropanel_col in range(pb):
                                for micropanel_row in range(tile_m):
                                    packed_a[panel_row_start + micropanel_row, micropanel_col] = \
                                        a[i + micropanel_row + panel_row_start, p + micropanel_col]
                        if mr > 0:
                            remain_start_row = mp * tile_m
                            for remain_col in range(pb):
                                for remain_row in range(mr):
                                    packed_a[remain_start_row + remain_row, remain_col] = \
                                        a[i + remain_start_row + remain_row, p + remain_col]
                                remain_row = mr
                                while remain_row < tile_m:
                                    packed_a[remain_start_row + remain_row, remain_col] = 0.0
                                    remain_row += 1

                        for nb in range(nbs):
                            j = nb * block_n
                            jb = min(block_n, n_size - j)
                            np = jb // tile_n
                            nr = jb % tile_n
                            for micropanel_idx in range(np):
                                panel_col_start = micropanel_idx * tile_n
                                for micropanel_row in range(pb):
                                    for micropanel_col in range(tile_n):
                                        packed_b[micropanel_row, micropanel_col + panel_col_start] = \
                                            b[p + micropanel_row, j + micropanel_col + panel_col_start]
                            if nr > 0:
                                remain_col_start = np * tile_n
                                for remain_row in range(pb):
                                    for remain_col in range(nr):
                                        packed_b[remain_row, remain_col + remain_col_start] = \
                                            b[p + remain_row, j + remain_col + remain_col_start]
                                    remain_col = nr
                                    while remain_col < tile_n:
                                        packed_b[remain_row, remain_col_start + remain_col] = 0.0
                                        remain_col += 1
                            macro_kernel(packed_a, packed_b, ~c[i, j], ib, jb, pb)
        assert isinstance(matmul_kernel_x86, hidet.ir.Function)
        matmul_kernel_x86.kind = "host_kernel"
        ir_module = module.ir_module()
        return ir_module


class Matmulx86Op(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        if not (len(a.shape) == len(b.shape) == 2 and a.shape[1] == b.shape[0]):
            raise ValueError(
                'Matrix multiplication: incompatible sizes: {} and {}'.format(
                    a.shape, b.shape
                )
            )
        task = MatmulF32Taskx86(input_like(a, 'a'), input_like(b, 'b'))
        super().__init__(inputs=[a, b], attributes={}, task=task)


def matmul_x86(a: Tensor, b: Tensor) -> Tensor:
    return Matmulx86Op(a, b).get_output(0)
