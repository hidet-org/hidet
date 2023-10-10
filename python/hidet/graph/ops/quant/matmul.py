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
from typing import List, Tuple, Optional
import hidet
from hidet.ir import dtypes
from hidet.ir.dtypes import float16, int8
from hidet.ir.expr import if_then_else, Int, Expr, cast
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.compute import TensorNode
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.utils import input_like, broadcast_shape, can_mutually_broadcast
from hidet.ir.library import tune
from hidet.graph.operator import Operator, Tensor
from hidet.utils.py import is_power_of_two, cdiv, prod
from hidet.graph.ops.utils import broadcast_indices
from hidet.graph.transforms import ResolveRule, register_resolve_rule
from hidet.ir.expr import is_constant


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


class SymmetricQuantizedMatmulF16Task(Task):
    def __init__(self, a: TensorNode, weight: TensorNode, scale: TensorNode, parallel_k_parts: int = 1):

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
        c_shape = [parallel_k_parts] + broadcast_shape(a.shape[:-2], weight.shape[:-2]) + [a_shape[-2], b_shape[-1]]
        k_part_extent = cdiv(k_size, parallel_k_parts)

        c = compute(
            name='c',
            shape=c_shape,
            fcompute=lambda k_part, *indices: reduce(
                shape=[k_part_extent],
                fcompute=lambda k: if_then_else(
                    k_part * k_part_extent + k < k_size,
                    a[broadcast_indices(indices[:-2], a.shape[:-2], c_shape[1:-2]) + [indices[-2], k]]
                    * (
                        cast(
                            weight[
                                broadcast_indices(indices[:-2], weight.shape[:-2], c_shape[1:-2]) + [k, indices[-1]]
                            ],
                            float16,
                        )
                        * scale[indices[-1]]
                    ),
                    float16(0.0),
                ),
                reduce_type='sum',
            ),
        )

        super().__init__(
            name='symmetric_quantized_matmulf16',
            inputs=[a, weight, scale],
            outputs=[c],
            attributes={'parallel_k_parts': parallel_k_parts},
        )

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> List[IRModule]:
        return tune.extract_ir_modules(self.schedule)

    def get_alignment(self, last_dim: int) -> int:
        if last_dim % 16 == 0:
            return 16
        elif last_dim % 8 == 0:
            return 8
        elif last_dim % 4 == 0:
            return 4
        else:
            return 1

    @tune.space(
        2,
        block_m=[32, 64, 128, 256],
        block_n=[64, 128, 256, 512],
        block_k=[16, 32, 64, 128],
        warp_m=[16, 32, 48, 64],
        warp_n=[16, 32, 48, 64, 96],
        warp_k=[8, 16, 32, 64],
        n_stages=[2, 3, 4, 5],
        mma=['m16n8k16'],
    )
    @tune.space(
        1,
        block_m=[128],
        block_n=[128],
        block_k=[16],
        warp_m=[64],
        warp_n=[64],
        warp_k=[16],
        n_stages=[2, 3],
        mma=['m16n8k16'],
    )
    def schedule(
        self, block_m=64, block_n=128, block_k=16, warp_m=32, warp_n=64, warp_k=16, n_stages=2, mma: str = 'm16n8k16'
    ) -> IRModule:
        # pylint: disable=unused-variable
        from hidet.ir.mapping import row_spatial, col_spatial, row_repeat
        from hidet.ir.type import tensor_type
        from hidet.lang import attrs, view, u32, f16, i32, tensor_pointer, grid
        from hidet.ir.stmt import asm
        from hidet.lang.layout import row_major
        from hidet.lang.mapping import spatial, auto_map
        from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory
        from hidet.lang.cuda import MmaConfig, mma_sync, cp_async, ldmatrix, cp_async_commit_group, cp_async_wait_group
        from hidet.lang.cuda import register_tensor

        # load a as usual, [block_m, block_k] of fp16 values into smem
        # each warp computes a fragment of [16, 16] or [16, 8] in shape, of fp16 values in registers

        # b is of type int8, we load a block of [block_k, block_n] of int8 values into smem
        # each warp loads a block of int8[16, 16] reinterpreted as fp16[16, 8] using ldmatrix instruction
        # each thread loads 8 int8 values from smem into registers, we then cast them to fp16 using scale

        # input shapes
        node_a, weight, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape: Tuple[Int, ...] = node_a.shape
        b_shape: Tuple[Int, ...] = weight.shape
        c_shape: Tuple[Int, ...] = node_c.shape
        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]
        a_head, b_head, c_head = list(a_shape[:-2]), list(b_shape[:-2]), list(c_shape[:-2])
        k_parts = self.attrs['parallel_k_parts']
        k_part_extent = cdiv(cdiv(k_size, k_parts), 8) * 8

        # schedule parameters
        mma_configs = {'m16n8k8': MmaConfig.m16n8k8_f16_f16(), 'm16n8k16': MmaConfig.m16n8k16_f16_f16()}
        tune.check(mma in mma_configs)
        mma_config = mma_configs[mma]

        mma_m = mma_config.m
        mma_n = mma_config.n * 2
        mma_k = mma_config.k

        warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
        mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
        threads = warp_count_m * warp_count_n * warp_count_k * 32
        grid_dim: Tuple[Int, Int, Int] = cdiv(m_size, block_m), cdiv(n_size, block_n), prod(c_head)
        # tile_a of size fp16[2, block_m, block_k] takes 2 * block_m * block_k * 2 bytes
        # tile_b of size int8[2, block_k, block_n] takes block_k * block_n bytes
        # tile_c of size fp16[block_m, block_n] takes block_m * block_n * 2 bytes
        # scale_parameters of size fp16[block_n] takes block_n * 2 bytes
        dynamic_smem_bytes = (
            max(n_stages * block_m * block_k * 2 + n_stages * block_n * block_k, block_m * block_n * 2) + block_n * 2
        )

        tune.check(block_m % warp_m == block_n % warp_n == block_k % warp_k == 0, 'warp dims divide block dims')
        tune.check(warp_m % mma_m == warp_n % mma_n == warp_k % mma_k == 0, 'mma dims divide warp dims')
        tune.check(threads <= 1024, 'threads in a block <= 1024')
        maximum_smem_bytes = 49152
        tune.check(dynamic_smem_bytes <= maximum_smem_bytes, 'dynamic shared memory <= 49152')

        tune.check(block_n % 128 == 0, 'block_n must be multiple of 128, required by async gmem -> smem loading')
        tune.check(block_k % 8 == 0)
        tune.check(is_power_of_two(block_k // 8))
        smem_a_type = tensor_type(
            'float16', shape=[block_m, block_k], layout=row_major(block_m, block_k // 8).swizzle(1) * row_major(1, 8)
        )
        smem_b_type = tensor_type(
            'int8',
            shape=[block_k, block_n],
            layout=row_major(block_k // 8, block_n // 128) * row_major(8, 8).swizzle(1) * row_major(1, 16),
        )

        load_smem_a_map = auto_map(block_m, block_k // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg))
        load_smem_b_map = auto_map(block_k, block_n // 16, workers=threads, on_fail=lambda msg: tune.check(False, msg))
        store_smem_c_map = auto_map(block_m, block_n, workers=threads, on_fail=lambda msg: tune.check(False, msg))

        with hidet.script_module() as module:

            @hidet.script
            def cvt_i8x4_f16x4(x: int8[4], y: f16[4]):
                xi = view(x, u32[1])
                yi = view(y, u32[2])

                asm("prmt.b32 %0,%1,%1,%2;", inputs=[xi[0], 0x9180], outputs=[yi[0]])
                asm("prmt.b32 %0,%1,%1,%2;", inputs=[xi[0], 0xB3A2], outputs=[yi[1]])

                asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[0], 0x03FF03FF, 0x66006600, 106], outputs=[yi[0]])
                asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[1], 0x03FF03FF, 0x66006600, 106], outputs=[yi[1]])

                rk = 0x66006600
                asm("sub.f16x2 %0, %1, %2;", inputs=[yi[0], rk], outputs=[yi[0]])
                asm("sub.f16x2 %0, %1, %2;", inputs=[yi[1], rk], outputs=[yi[1]])

            @hidet.script
            def load_smem_fixed(smem: ~int8, gmem: ~int8, alignment: i32, target_bytes: i32, src_bytes: i32):
                """
                Loads src_bytes from gmem into smem

                alignment specifies the cp_async instruction used, while target_bytes specifies the size
                of smem mutated, starting from the address provided, it must be a multiple of alignment

                This parameter is used as cp_async writes zeros to all bytes from smem[src_bytes:target_bytes]
                """

                if alignment == 16:
                    for i in range(cdiv(target_bytes, 16)):
                        cp_async(~smem[i * 16], ~gmem[i * 16], cp_size=16, src_size=src_bytes, cache_level='global')
                        src_bytes = max(0, src_bytes - 16)
                elif alignment == 8:
                    for i in range(cdiv(target_bytes, 8)):
                        cp_async(~smem[i * 8], ~gmem[i * 8], cp_size=8, src_size=src_bytes)
                        src_bytes = max(0, src_bytes - 8)
                elif alignment == 4:
                    for i in range(cdiv(target_bytes, 4)):
                        cp_async(~smem[i * 4], ~gmem[i * 4], cp_size=4, src_size=src_bytes)
                        src_bytes = max(0, src_bytes - 4)
                else:  # slow fallback
                    for i in range(src_bytes):
                        smem[i] = gmem[i]

            @hidet.script
            def load_regs_a(mi: int, k1: int, smem_a: smem_a_type, regs_a: float16[mma_config.a_elements]):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    row_addr = ~smem_a[wi * warp_m + mi * mma_m + p, wk * warp_k + k1 * mma_k + q * 8]
                    b32_regs = view(regs_a, u32[4])
                    ldmatrix(
                        regs=[b32_regs[0], b32_regs[1], b32_regs[2], b32_regs[3]],
                        smem_addr=row_addr,
                        shared_space_addr=False,
                        trans=False,
                    )

            @hidet.script
            def load_regs_b(
                mj: int,
                k1: int,
                smem_b: smem_b_type,
                regs_b: float16[2, mma_config.b_elements],
                scale: float16[block_n],
            ):
                # smem_b_type = float16[block_k, block_n]
                # smem_scale_type = float16[block_n]

                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    # have not used q as we only use the address of the first 16 threads to load 2 of 8x8 f16 matrix.

                    offset_block_n = wj * warp_n + mj * mma_n
                    row_addr = ~smem_b[wk * warp_k + k1 * mma_k + p, offset_block_n]
                    temp_regs = register_tensor(u32, shape=[2])
                    ldmatrix(regs=[temp_regs[0], temp_regs[1]], smem_addr=row_addr, trans=True)

                    regs = view(~temp_regs, int8[8])
                    regs_new = register_tensor(int8, shape=[2, 4])

                    for i in range(8):
                        regs_new[i % 2, i // 2] = regs[i]

                    cvt_i8x4_f16x4(~regs_new[0, 0], ~regs_b[0, 0])
                    cvt_i8x4_f16x4(~regs_new[1, 0], ~regs_b[1, 0])

            @hidet.script
            def warp_mma(
                regs_a: float16[mma_config.a_elements],
                regs_b: float16[2, mma_config.b_elements],
                regs_c: float16[2, mma_config.c_elements],
            ):
                mma_sync(mma_config, regs_a, ~regs_b[0, 0], ~regs_c[0, 0])
                mma_sync(mma_config, regs_a, ~regs_b[1, 0], ~regs_c[1, 0])

            @hidet.script
            def load_smem_a(k0: int, a: float16[a_head + [m_size, k_size]], smem_a: smem_a_type):
                c_head_index = spatial(*c_head).map(blockIdx.z)
                offset_m = blockIdx.x * block_m
                offset_k = c_head_index[0] * k_part_extent + k0 * block_k
                maximum_k = min(k_size, (c_head_index[0] + 1) * k_part_extent)
                gmem_a = a[broadcast_indices(c_head_index[1:], a_head, c_head[1:])][offset_m:, offset_k:]
                for i, k_seg in load_smem_a_map.on(threadIdx.x):
                    k = k_seg * 8
                    src_size = (
                        0
                        if (offset_m + i >= m_size or offset_k + k >= maximum_k)
                        else min(maximum_k - (offset_k + k), 8)
                    )
                    alignment = self.get_alignment(k_size)
                    load_smem_fixed(cast(~smem_a[i, k], ~int8), cast(~gmem_a[i, k], ~int8), alignment, 16, src_size * 2)

            @hidet.script
            def load_smem_b(k0: int, b: int8[b_head + [k_size, n_size]], smem_b: smem_b_type):
                c_head_index = spatial(*c_head).map(blockIdx.z)
                offset_n = blockIdx.y * block_n
                offset_k = c_head_index[0] * k_part_extent + k0 * block_k
                maximum_k = min(k_size, (c_head_index[0] + 1) * k_part_extent)
                gmem_b = b[broadcast_indices(c_head_index[1:], b_head, c_head[1:])][offset_k:, offset_n:]
                for k, j_seg in load_smem_b_map.on(threadIdx.x):
                    j = j_seg * 16
                    src_size = (
                        0 if (offset_k + k >= maximum_k or offset_n + j >= n_size) else min(n_size - (offset_n + j), 16)
                    )
                    alignment = self.get_alignment(n_size)
                    # the target_cp_size is 16 because ld_matrix requires 16-byte alignment
                    load_smem_fixed(~smem_b[k, j], ~gmem_b[k, j], alignment, 16, src_size)

            @hidet.script
            def matmul_f16_i8_kernelv2(
                a: float16[a_head + [m_size, k_size]],
                b: int8[b_head + [k_size, n_size]],
                scale: float16[n_size],
                c: float16[c_head + [m_size, n_size]],
            ):
                # matrix multiplication, using mma instruction
                attrs.cuda.grid_dim = grid_dim
                attrs.cuda.block_dim = threads
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes
                # smem_storage = dyn_smem_storage
                smem_a = tensor_pointer(
                    'float16', shape=[n_stages, block_m, block_k], layout=row_major(n_stages) + smem_a_type.layout
                )
                smem_b = tensor_pointer(
                    'int8', shape=[n_stages, block_k, block_n], layout=row_major(n_stages) + smem_b_type.layout
                )
                smem_scale = tensor_pointer('float16', shape=[block_n])  # we use row_major for now

                smem_a = dynamic_shared_memory(byte_offset=0, dtype=float16)
                smem_b = dynamic_shared_memory(byte_offset=n_stages * block_m * block_k * 2, dtype=int8)
                smem_scale = dynamic_shared_memory(
                    byte_offset=n_stages * block_m * block_k * 2 + n_stages * block_k * block_n, dtype=float16
                )

                regs_a = register_tensor(float16, [2, mma_count_m, mma_config.a_elements])
                regs_b = register_tensor(float16, [2, mma_count_n, 2, mma_config.b_elements])
                regs_c = register_tensor(float16, [mma_count_m, mma_count_n, 2, mma_config.c_elements])

                for i, j, k, p in grid(mma_count_m, mma_count_n, 2, mma_config.c_elements):
                    regs_c[i, j, k, p] = 0.0

                # for i in load_smem_scale_map.on(threadIdx.x):
                #     if i < block_n:
                #         if i + blockIdx.y * block_n < n_size:
                #             smem_scale[i] = scale[i + blockIdx.y * block_n]
                #         else:
                #             smem_scale[i] = 1.0  # identity w.r.t. multiplication

                if threads * 8 > block_n:
                    offset = threadIdx.x * 8 + blockIdx.y * block_n
                    if offset < n_size and threadIdx.x * 8 < block_n:
                        src_size = min(8, n_size - offset)
                        load_smem_fixed(
                            cast(~smem_scale[threadIdx.x * 8], ~int8), cast(~scale[offset], ~int8), 16, 16, src_size * 2
                        )
                else:
                    loc_tid = threadIdx.x * 8
                    while loc_tid < block_n:
                        offset = loc_tid + blockIdx.y * block_n
                        if offset < n_size and loc_tid < block_n:
                            src_size = min(8, n_size - offset)
                            load_smem_fixed(
                                cast(~smem_scale[loc_tid], ~int8), cast(~scale[offset], ~int8), 16, 16, src_size * 2
                            )
                            loc_tid += threads * 8

                for ki in range(n_stages - 1):
                    load_smem_a(ki, a, ~smem_a[ki, 0, 0])
                    load_smem_b(ki, b, ~smem_b[ki, 0, 0])
                    cp_async_commit_group()

                cp_async_wait_group(n_stages - 2)
                syncthreads()

                load_kn = (k_part_extent + block_k - 1) // block_k
                for k0 in range(load_kn):
                    for mi in range(mma_count_m):
                        load_regs_a(mi, 0, ~smem_a[k0 % n_stages, 0, 0], ~regs_a[0, mi, 0])
                    for mj in range(mma_count_n):
                        load_regs_b(mj, 0, ~smem_b[k0 % n_stages, 0, 0], ~regs_b[0, mj, 0, 0], ~smem_scale[0])

                    for mk in range(mma_count_k):
                        if mk + 1 < mma_count_k:
                            for mi in range(mma_count_m):
                                load_regs_a(mi, mk + 1, ~smem_a[k0 % n_stages, 0, 0], ~regs_a[(mk + 1) % 2, mi, 0])
                            for mj in range(mma_count_n):
                                load_regs_b(
                                    mj,
                                    mk + 1,
                                    ~smem_b[k0 % n_stages, 0, 0],
                                    ~regs_b[(mk + 1) % 2, mj, 0, 0],
                                    ~smem_scale[0],
                                )

                        for mi, mj in grid(mma_count_m, mma_count_n):
                            warp_mma(~regs_a[mk % 2, mi, 0], ~regs_b[mk % 2, mj, 0, 0], ~regs_c[mi, mj, 0, 0])

                    load_smem_a(k0 + n_stages - 1, a, ~smem_a[(k0 + n_stages - 1) % n_stages, 0, 0])
                    load_smem_b(k0 + n_stages - 1, b, ~smem_b[(k0 + n_stages - 1) % n_stages, 0, 0])
                    cp_async_commit_group()
                    cp_async_wait_group(n_stages - 2)
                    syncthreads()

                # store back
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
                c_head_index = spatial(*c_head).map(blockIdx.z)
                gmem_c = c[c_head_index][offset_m:, offset_n:]

                c_store_map = row_repeat(2, 1, attrs='u+u+') * row_spatial(8, 4) * row_repeat(1, 4, attrs='u+u+')

                if warp_count_k == 1:
                    for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            p = 0
                            for i, j in c_store_map.on(lane_id):
                                delta_m = wi * warp_m + mi * mma_m + i
                                delta_n = wj * warp_n + mj * mma_n + j
                                in_bound = (offset_m + delta_m < m_size) and (offset_n + delta_n < n_size)
                                if in_bound:
                                    gmem_c[delta_m, delta_n] = regs_c[mi, mj, p % 2, p // 2] * smem_scale[delta_n]
                                p += 1
                else:
                    smem_c = tensor_pointer('float16', shape=[block_m, block_n])
                    smem_c = dynamic_shared_memory(byte_offset=0, dtype=float16)

                    for k_round in range(warp_count_k):
                        for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                            if wk == k_round:
                                for mi, mj in grid(mma_count_m, mma_count_n):
                                    p = 0
                                    for i, j in c_store_map.on(lane_id):
                                        delta_m = wi * warp_m + mi * mma_m + i
                                        delta_n = wj * warp_n + mj * mma_n + j
                                        in_bound = (offset_m + delta_m < m_size) and (offset_n + delta_n < n_size)
                                        if in_bound:
                                            if k_round == 0:
                                                smem_c[delta_m, delta_n] = regs_c[mi, mj, p % 2, p // 2]
                                            else:
                                                smem_c[delta_m, delta_n] += regs_c[mi, mj, p % 2, p // 2]
                                        p += 1
                        if warp_count_k > 1:
                            syncthreads()
                    for i, j in store_smem_c_map.on(threadIdx.x):
                        if offset_m + i < m_size and offset_n + j < n_size:
                            gmem_c[i, j] = smem_c[i, j] * smem_scale[j]

        ir_module = module.ir_module()
        assert isinstance(matmul_f16_i8_kernelv2, Function)

        return ir_module


class SymmetricQuantizedMatmulF16Op(Operator):
    def __init__(self, a: Tensor, b: Tensor, scale: Tensor, parallel_k_parts=1):
        if not (isinstance(parallel_k_parts, int) and not isinstance(parallel_k_parts, bool)):
            raise ValueError('parallel_k_parts must be an integer, got {}'.format(parallel_k_parts))
        super().__init__(
            inputs=[a, b, scale],
            attributes={'parallel_k_parts': parallel_k_parts},
            task=SymmetricQuantizedMatmulF16Task(
                input_like(a, 'a'), input_like(b, 'b'), input_like(scale, 'scale'), parallel_k_parts
            ),
        )


def symmetric_quant_matmulf16(a: Tensor, weight: Tensor, scale: Tensor, parallel_k_parts=1) -> Tensor:
    if len(a.shape) < 2 or len(weight.shape) < 2:
        raise ValueError('a and b must have at least 2 dimensions, got shape {} and {}'.format(a.shape, weight.shape))
    # TODO: impliment dynamic run-time shape assertion
    if not (isinstance(a.shape[-1], Expr) or isinstance(weight.shape[-1], Expr)) and (
        a.shape[-1] % 2 != 0 or weight.shape[-1] % 2 != 0
    ):
        raise ValueError('Expect the last dimension of the input tensors to be a multiple of 2')
    if a.dtype != dtypes.float16 or weight.dtype != dtypes.int8:
        raise ValueError('BatchMatmulF16Op only support float16, int8, got {} and {}'.format(a.dtype, weight.dtype))
    return SymmetricQuantizedMatmulF16Op(a, weight, scale, parallel_k_parts).outputs[0]


@register_resolve_rule(SymmetricQuantizedMatmulOp)
class QuantSymmetricResolveRule(ResolveRule):
    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        # if op.task.has_symbolic_shape():
        #     return None

        a: Tensor = op.inputs[0]
        b: Tensor = op.inputs[1]
        scale: Tensor = op.inputs[2]
        c: Tensor = op.outputs[0]

        if hidet.option.cuda.get_arch_pair() < (8, 0):
            return None

        parallel_k = self.get_config('parallel_k', default='default')  # 'default', 'search', 2, 4, ...

        if not (is_constant(a.shape[-1]) and is_constant(b.shape[-2])):
            k_parts = 1
        elif isinstance(parallel_k, str):
            if parallel_k == 'default':
                batch_size, m_size, n_size, k_size = prod(c.shape[:-2]), c.shape[-2], c.shape[-1], a.shape[-1]
                if is_constant(batch_size, m_size):
                    estimate_blocks = batch_size * cdiv(m_size, 64) * cdiv(n_size, 64)
                    estimate_concurrent_blocks = 80 * 5
                    max_k_parts = cdiv(k_size, 64)
                    k_parts = min(cdiv(estimate_concurrent_blocks, estimate_blocks), max_k_parts)
                else:
                    k_parts = 1
            elif parallel_k == 'disabled':
                k_parts = 1
            elif parallel_k == 'search':
                candidates = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
                # to get around vcuda error
                # temporary hack to sample latency from symbolic shape
                a_shape = list(a.shape)
                b_shape = list(b.shape)
                for i in range(len(a_shape)):
                    if not is_constant(a_shape[i]):
                        a_shape[i] = 1
                for i in range(len(b_shape)):
                    if not is_constant(b_shape[i]):
                        b_shape[i] = 1

                aa = hidet.symbol(a_shape, dtype=a.dtype, device='cuda')
                bb = hidet.symbol(b_shape, dtype=b.dtype, device='cuda')
                sscale = hidet.symbol_like(scale, device='cuda')

                latencies: List[float] = []
                print(
                    'Symmetric Quantized Matmul: Searching the best parallel_k for {} x {} x {} among {}'.format(
                        a.shape, b.shape, scale.shape, candidates
                    )
                )
                for candidate in candidates:
                    cc = symmetric_quant_matmulf16(aa, bb, sscale, parallel_k_parts=candidate)
                    cc = cc.sum(0)
                    graph = hidet.trace_from([cc], [aa, bb, sscale])
                    # prevent recursion
                    with hidet.graph.PassContext():
                        graph: hidet.FlowGraph = hidet.graph.optimize(graph)
                    latency: float = graph.latency()
                    latencies.append(latency)
                best_idx = min(range(len(candidates)), key=lambda i: latencies[i])
                print(
                    'Results: {{{}}},'.format(
                        ', '.join('{}: {:.1f}'.format(a, b * 1000) for a, b in zip(candidates, latencies))
                    ),
                    'Picked {} with {:.1f} micro-seconds'.format(candidates[best_idx], latencies[best_idx] * 1000),
                )
                k_parts = candidates[best_idx]
            else:
                raise ValueError(f'invalid parallel_k: {parallel_k}')
        elif isinstance(parallel_k, int):
            k_parts = min(max(parallel_k, 1), 32)
        else:
            raise ValueError(f'invalid parallel_k: {parallel_k}')
        c = symmetric_quant_matmulf16(a, b, scale, parallel_k_parts=k_parts).sum(0)
        return [c]
