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
from functools import partial

from hidet.ir import dtypes
from hidet.ir.type import DataType, data_type
from hidet.ir.dtypes import float16, u64
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

from hidet import option


class MatmulF16SM90Task(Task):
    def __init__(
        self,
        a: TensorNode,
        b: TensorNode,
        is_a_shared: bool,
        acc_dtype: Union[DataType, str],
        parallel_k_parts: int = 1,
    ):
        if not a.type.dtype == float16 or not b.type.dtype == float16:
            raise ValueError('Both inputs must be float16 tensors')

        if len(a.shape) < 2 or len(b.shape) < 2:
            raise ValueError('Matrix multiplication expect at least 2D tensor, got {} and {}'.format(a.shape, b.shape))

        self._assert(
            a.shape[-1] == b.shape[-2],
            msg=(
                'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                ', got {} and {}'.format(a.shape, b.shape)
            ),
        )

        self._assert(
            can_mutually_broadcast(a.shape[:-2], b.shape[:-2]),
            msg=(
                'Matrix multiplication expect tensor A and B with compatible broadcast shape, '
                'got {} and {}'.format(a.shape, b.shape)
            ),
        )

        a_shape = a.shape
        b_shape = b.shape
        k_size = a.shape[-1]
        c_shape = [parallel_k_parts] + broadcast_shape(a.shape[:-2], b.shape[:-2]) + [a_shape[-2], b_shape[-1]]
        k_part_extent = cdiv(k_size, parallel_k_parts)
        acc_dtype = data_type(acc_dtype)

        def inner_compute(k_part, indices, k):
            return if_then_else(
                k_part * k_part_extent + k < k_size,
                cast(
                    a[broadcast_indices(indices[:-2], a.shape[:-2], c_shape[1:-2]) + [indices[-2], k]]
                    * b[broadcast_indices(indices[:-2], b.shape[:-2], c_shape[1:-2]) + [k, indices[-1]]],
                    acc_dtype,
                ),
                acc_dtype(0.0),
            )

        def outer_compute(k_part, *indices):
            return float16(
                reduce(shape=[k_part_extent], fcompute=partial(inner_compute, k_part, indices), reduce_type='sum')
            )

        c = compute(name='c', shape=c_shape, fcompute=outer_compute)

        super().__init__(
            name='matmul_f16_sm90',
            inputs=[a, b],
            outputs=[c],
            attributes={'acc_dtype': acc_dtype, 'parallel_k_parts': parallel_k_parts, 'is_a_shared': is_a_shared},
        )

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> List[IRModule]:
        # FIXME: currently, we need to disable multi-processing
        # while generating the IR modules for Hopper kernels
        # because it does not work with the on-demand registration
        # introduced in the PR #677
        with option.context():
            option.num_local_workers(1)
            return tune.extract_ir_modules(self.schedule)

    @tune.space(
        2,
        block_m=[64, 128, 256],
        block_n=[64, 128, 256],
        block_k=[16, 32, 64, 128],
        warp_group_m=[64, 128],
        warp_group_n=[16, 32, 64, 128, 256],
        warp_group_k=[16, 32, 64],
    )
    @tune.space(
        1, block_m=[64, 128], block_n=[128], block_k=[16], warp_group_m=[64], warp_group_n=[64, 128], warp_group_k=[16]
    )
    def schedule(
        self, block_m=64, block_n=128, block_k=32, warp_group_m=64, warp_group_n=128, warp_group_k=32
    ) -> IRModule:
        # pylint: disable=unused-variable
        import hidet
        from hidet.ir.type import tensor_type
        from hidet.lang import attrs, col_spatial, view, u32, tensor_pointer, grid
        from hidet.lang.layout import row_major, column_major
        from hidet.lang.mapping import spatial, auto_map
        from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory
        from hidet.lang.cuda import cp_async, cp_async_wait_all, ldmatrix
        from hidet.lang.cuda import register_tensor
        from hidet.lang.cuda import (
            WgmmaConfig,
            wgmma_async,
            make_wgmma_desc,
            wgmma_wait_group,
            wgmma_commit_group,
            wgmma_fence,
        )
        from hidet.ir.primitives.cuda.barrier import fence_view_async_shared
        from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

        # input shapes
        node_a, node_b, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape: Tuple[Int, ...] = node_a.shape
        b_shape: Tuple[Int, ...] = node_b.shape
        c_shape: Tuple[Int, ...] = node_c.shape
        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]
        a_head, b_head, c_head = list(a_shape[:-2]), list(b_shape[:-2]), list(c_shape[:-2])
        k_parts = self.attrs['parallel_k_parts']
        k_part_extent = cdiv(cdiv(k_size, k_parts), 8) * 8
        acc_dtype = self.attrs['acc_dtype']

        # schedule parameters
        mma_configs_f16 = {f"m64n{n}k16": WgmmaConfig.get(64, n, 16, "f16", "f16", "f16") for n in range(8, 257, 8)}
        mma_configs_f32 = {f"m64n{n}k16": WgmmaConfig.get(64, n, 16, "f16", "f16", "f32") for n in range(8, 257, 8)}
        if warp_group_n > 256:
            mma = 'm64n256k16'
        else:
            mma = f'm64n{warp_group_n}k16'

        tune.check(mma in mma_configs_f16 or mma in mma_configs_f32)
        mma_config = mma_configs_f16[mma] if acc_dtype == float16 else mma_configs_f32[mma]

        mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 64, warp_group_n, 16

        warp_group_count_m, warp_group_count_n, warp_group_count_k = (
            block_m // warp_group_m,
            block_n // warp_group_n,
            block_k // warp_group_k,
        )

        mma_count_m, mma_count_n, mma_count_k = warp_group_m // mma_m, warp_group_n // mma_n, warp_group_k // mma_k
        threads = warp_group_count_m * warp_group_count_n * warp_group_count_k * 128  # warp group
        grid_dim: Tuple[Int, Int, Int] = cdiv(m_size, block_m), cdiv(n_size, block_n), prod(c_head)
        dynamic_smem_bytes = max(2 * (block_m + block_n) * block_k * 2, block_m * block_n * acc_dtype.nbytes)
        tune.check(
            block_m % warp_group_m == block_n % warp_group_n == block_k % warp_group_k == 0,
            'warp dims divide block dims',
        )
        tune.check(
            warp_group_m % mma_m == warp_group_n % mma_n == warp_group_k % mma_k == 0, 'mma dims divide warp dims'
        )
        tune.check(
            (mma_n == 256 and threads <= 256) or (mma_n != 256 and threads <= 512),
            'Invalid configuration: threads must be <= 256 if mma_n == 256, else threads must be <= 512',
        )
        maximum_smem_bytes = 227 * 1024
        tune.check(dynamic_smem_bytes <= maximum_smem_bytes, 'dynamic shared memory <= 227 * 1024')

        tune.check(block_n % 64 == 0, 'block_n must be multiple of 64, required by async gmem -> smem loading')
        tune.check(block_k % 8 == 0)
        tune.check(is_power_of_two(block_k // 8))

        if not self.attrs['is_a_shared']:
            smem_a_type = tensor_type(
                'float16',
                shape=[block_m, block_k],
                layout=row_major(block_m, block_k // 8).swizzle(1) * row_major(1, 8),
            )
        else:
            if block_k % 64 == 0:
                smem_a_type = tensor_type(
                    'float16',
                    shape=[block_m, block_k],
                    layout=row_major(block_m // 8, block_k // 64)
                    * row_major(8, 8).swizzle(dim=1, regards_dim=0, log_step=0)
                    * row_major(1, 8),
                )
            elif block_k % 32 == 0:
                smem_a_type = tensor_type(
                    'float16',
                    shape=[block_m, block_k],
                    layout=row_major(block_m // 8, block_k // 32)
                    * row_major(8, 4).swizzle(dim=1, regards_dim=0, log_step=1)
                    * row_major(1, 8),
                )
            elif block_k % 16 == 0:
                smem_a_type = tensor_type(
                    'float16',
                    shape=[block_m, block_k],
                    layout=row_major(block_m // 8, block_k // 16)
                    * row_major(8, 2).swizzle(dim=1, regards_dim=0, log_step=2)
                    * row_major(1, 8),
                )
            else:
                smem_a_type = tensor_type(
                    'float16', shape=[block_m, block_k], layout=row_major(block_m // 8, block_k // 8) * row_major(8, 8)
                )

        if block_n % 64 == 0:
            smem_b_type = tensor_type(
                'float16',
                shape=[block_k, block_n],
                layout=column_major(block_k // 8, block_n // 64)
                * row_major(8, 8).swizzle(dim=1, regards_dim=0, log_step=0)
                * row_major(1, 8),
            )
        elif block_n % 32 == 0:
            smem_b_type = tensor_type(
                'float16',
                shape=[block_k, block_n],
                layout=column_major(block_k // 8, block_n // 32)
                * row_major(8, 4).swizzle(dim=1, regards_dim=0, log_step=1)
                * row_major(1, 8),
            )
        elif block_n % 16 == 0:
            smem_b_type = tensor_type(
                'float16',
                shape=[block_k, block_n],
                layout=column_major(block_k // 8, block_n // 16)
                * row_major(8, 2).swizzle(dim=1, regards_dim=0, log_step=2)
                * row_major(1, 8),
            )
        else:
            smem_b_type = tensor_type(
                'float16', shape=[block_k, block_n], layout=column_major(block_k // 8, block_n // 8) * row_major(8, 8)
            )

        load_smem_a_map = auto_map(block_m, block_k // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg))
        load_smem_b_map = auto_map(block_k, block_n // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg))
        store_smem_c_map = auto_map(block_m, block_n, workers=threads, on_fail=lambda msg: tune.check(False, msg))

        with hidet.script_module() as module:

            @hidet.script
            def load_regs_a(mi: int, k1: int, smem_a: smem_a_type, regs_a: float16[mma_config.a_elements]):
                warp_id_in_group, lane_id = (threadIdx.x // 32) % 4, threadIdx.x % 32
                warp_group_id = threadIdx.x // 128
                for wi, wj, wk in spatial(warp_group_count_m, warp_group_count_n, warp_group_count_k).on(warp_group_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    row_addr = ~smem_a[
                        wi * warp_group_m + mi * mma_m + warp_id_in_group * (mma_m // 4) + p,
                        wk * warp_group_k + k1 * mma_k + q * 8,
                    ]
                    b32_regs = view(regs_a, u32[4])
                    ldmatrix(
                        regs=[b32_regs[0], b32_regs[1], b32_regs[2], b32_regs[3]],
                        smem_addr=row_addr,
                        shared_space_addr=False,
                        trans=False,
                    )

            # when core matrix is 8 * 8 and column_major slide core matrix to fill the smem
            # The SBO block_k // 8 * 64 * 2 calculate the byte offset of
            # first element of a core matrix to first element of next core matrix in next column
            if block_n % 64 == 0:
                b_desc_template: u64 = make_wgmma_desc(block_k // 8 * 512 * 2, 1024, 1)
            elif block_n % 32 == 0:
                b_desc_template: u64 = make_wgmma_desc(block_k // 8 * 256 * 2, 512, 2)
            elif block_n % 16 == 0:
                b_desc_template: u64 = make_wgmma_desc(block_k // 8 * 128 * 2, 256, 3)
            else:
                b_desc_template: u64 = make_wgmma_desc(128, block_k // 8 * 64 * 2, 0)

            if block_k % 64 == 0:
                a_desc_template: u64 = make_wgmma_desc(1, block_k // 64 * 512 * 2, 1)
            elif block_k % 32 == 0:
                a_desc_template: u64 = make_wgmma_desc(1, block_k // 32 * 256 * 2, 2)
            elif block_k % 16 == 0:
                a_desc_template: u64 = make_wgmma_desc(1, block_k // 16 * 128 * 2, 3)
            else:
                a_desc_template: u64 = make_wgmma_desc(128, block_k // 8 * 64 * 2, 0)

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
                    if a_shape[-1] % 8 == 0:
                        cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=16, src_size=src_size * 2, cache_level='global')
                    # trivially support other cp_sizes, perhaps do this in a more clever way?
                    elif a_shape[-1] % 4 == 0:
                        cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=8, src_size=min(8, src_size * 2))
                        cp_async(~smem_a[i, k + 4], ~gmem_a[i, k + 4], cp_size=8, src_size=max(0, src_size * 2 - 8))
                    elif a_shape[-1] % 2 == 0:
                        cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=4, src_size=min(4, src_size * 2))
                        cp_async(
                            ~smem_a[i, k + 2], ~gmem_a[i, k + 2], cp_size=4, src_size=min(4, max(0, src_size * 2 - 4))
                        )
                        cp_async(
                            ~smem_a[i, k + 4], ~gmem_a[i, k + 4], cp_size=4, src_size=min(4, max(0, src_size * 2 - 8))
                        )
                        cp_async(
                            ~smem_a[i, k + 6], ~gmem_a[i, k + 6], cp_size=4, src_size=min(4, max(0, src_size * 2 - 12))
                        )

            @hidet.script
            def load_smem_b(k0: int, b: float16[b_head + [k_size, n_size]], smem_b: smem_b_type):
                c_head_index = spatial(*c_head).map(blockIdx.z)
                offset_n = blockIdx.y * block_n
                offset_k = c_head_index[0] * k_part_extent + k0 * block_k
                maximum_k = min(k_size, (c_head_index[0] + 1) * k_part_extent)
                gmem_b = b[broadcast_indices(c_head_index[1:], b_head, c_head[1:])][offset_k:, offset_n:]
                for k, j_seg in load_smem_b_map.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = (
                        0 if (offset_k + k >= maximum_k or offset_n + j >= n_size) else min(n_size - (offset_n + j), 8)
                    )
                    if b_shape[-1] % 8 == 0:
                        cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=16, src_size=src_size * 2, cache_level='global')
                    # trivially support other cp_sizes, perhaps do this in a more clever way?
                    elif b_shape[-1] % 4 == 0:
                        cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=8, src_size=min(8, src_size * 2))
                        cp_async(~smem_b[k, j + 4], ~gmem_b[k, j + 4], cp_size=8, src_size=max(0, src_size * 2 - 8))
                    elif b_shape[-1] % 2 == 0:
                        cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=4, src_size=min(4, src_size * 2))
                        cp_async(
                            ~smem_b[k, j + 2], ~gmem_b[k, j + 2], cp_size=4, src_size=min(4, max(0, src_size * 2 - 4))
                        )
                        cp_async(
                            ~smem_b[k, j + 4], ~gmem_b[k, j + 4], cp_size=4, src_size=min(4, max(0, src_size * 2 - 8))
                        )
                        cp_async(
                            ~smem_b[k, j + 6], ~gmem_b[k, j + 6], cp_size=4, src_size=min(4, max(0, src_size * 2 - 12))
                        )

            @hidet.script
            def matmul_f16_kernel(
                a: float16[a_head + [m_size, k_size]],
                b: float16[b_head + [k_size, n_size]],
                c: float16[c_head + [m_size, n_size]],
            ):
                # matrix multiplication, using mma instruction
                attrs.cuda.grid_dim = grid_dim
                attrs.cuda.block_dim = threads
                # the second 2 means '2 bytes per float16'
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes
                # smem_storage = dyn_smem_storage
                smem_a = tensor_pointer(
                    'float16', shape=[2, block_m, block_k], layout=row_major(2) + smem_a_type.layout
                )
                smem_b = tensor_pointer(
                    'float16', shape=[2, block_k, block_n], layout=row_major(2) + smem_b_type.layout
                )
                smem_a = dynamic_shared_memory(byte_offset=0, dtype=float16)
                smem_b = dynamic_shared_memory(byte_offset=2 * block_m * block_k * 2, dtype=float16)
                if not self.attrs['is_a_shared']:
                    regs_a = register_tensor(float16, [2, mma_count_m, mma_config.a_elements])
                regs_c = register_tensor(acc_dtype, [mma_count_m, mma_count_n, mma_config.c_elements])

                for i, j, p in grid(mma_count_m, mma_count_n, mma_config.c_elements):
                    regs_c[i, j, p] = 0.0

                load_smem_a(0, a, ~smem_a[0, 0, 0])
                load_smem_b(0, b, ~smem_b[0, 0, 0])
                cp_async_wait_all()

                syncthreads()
                for k0 in range((k_part_extent + block_k - 1) // block_k):
                    load_smem_a(k0 + 1, a, ~smem_a[(k0 + 1) % 2, 0, 0])
                    load_smem_b(k0 + 1, b, ~smem_b[(k0 + 1) % 2, 0, 0])
                    if not self.attrs['is_a_shared']:
                        for mi in range(mma_count_m):
                            load_regs_a(mi, 0, ~smem_a[k0 % 2, 0, 0], ~regs_a[0, mi, 0])

                    for mk in range(mma_count_k):
                        if not self.attrs['is_a_shared']:
                            if mk + 1 < mma_count_k:
                                for mi in range(mma_count_m):
                                    load_regs_a(mi, mk + 1, ~smem_a[k0 % 2, 0, 0], ~regs_a[(mk + 1) % 2, mi, 0])

                        for mi, mj in grid(mma_count_m, mma_count_n):
                            warp_group_id = threadIdx.x // 128

                            wi, wj, wk = spatial(warp_group_count_m, warp_group_count_n, warp_group_count_k).map(
                                warp_group_id
                            )
                            smem_b_addr = cvta_generic_to_shared(
                                ~smem_b[k0 % 2, wk * warp_group_k + mk * mma_k, wj * warp_group_n + mj * mma_n]
                            )
                            b_matrix_start_addr = (smem_b_addr & 0x3FFFF) >> 4
                            b_desc: u64 = b_desc_template | (b_matrix_start_addr) << 0
                            b_desc = b_desc | ((smem_b_addr >> 0x7) & 0x7) << 49
                            if self.attrs['is_a_shared']:
                                smem_a_addr = cvta_generic_to_shared(
                                    ~smem_a[k0 % 2, wi * warp_group_m + mi * mma_m, wk * warp_group_k + mk * mma_k]
                                )
                                a_matrix_start_addr = (smem_a_addr & 0x3FFFF) >> 4
                                a_desc: u64 = a_desc_template | (a_matrix_start_addr) << 0
                                a_desc = a_desc | ((smem_a_addr >> 0x7) & 0x7) << 49

                            wgmma_fence()
                            fence_view_async_shared()
                            if self.attrs['is_a_shared']:
                                wgmma_async(mma_config, a_desc, ~regs_c[mi, mj, 0], b_desc, trans_a=0, trans_b=1)
                            else:
                                wgmma_async(mma_config, ~regs_a[mk % 2, mi, 0], ~regs_c[mi, mj, 0], b_desc, trans_b=1)
                            wgmma_commit_group()
                    wgmma_wait_group(0)
                    cp_async_wait_all()
                    syncthreads()

                # store back
                lane_id = threadIdx.x % 128
                warp_group_id = threadIdx.x / 128
                offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
                c_head_index = spatial(*c_head).map(blockIdx.z)
                gmem_c = c[c_head_index][offset_m:, offset_n:]

                if warp_group_count_k == 1:
                    for wi, wj, wk in spatial(warp_group_count_m, warp_group_count_n, warp_group_count_k).on(
                        warp_group_id
                    ):
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            p = 0
                            for i, j in mma_config.c_store_map.on(lane_id):
                                delta_m = wi * warp_group_m + mi * mma_m + i
                                delta_n = wj * warp_group_n + mj * mma_n + j
                                in_bound = (offset_m + delta_m < m_size) and (offset_n + delta_n < n_size)
                                if in_bound:
                                    gmem_c[delta_m, delta_n] = float16(regs_c[mi, mj, p])
                                p += 1
                else:
                    smem_c = tensor_pointer(acc_dtype, shape=[block_m, block_n])
                    smem_c = dynamic_shared_memory(byte_offset=0, dtype=acc_dtype)

                    for k_round in range(warp_group_count_k):
                        for wi, wj, wk in spatial(warp_group_count_m, warp_group_count_n, warp_group_count_k).on(
                            warp_group_id
                        ):
                            if wk == k_round:
                                for mi, mj in grid(mma_count_m, mma_count_n):
                                    p = 0
                                    for i, j in mma_config.c_store_map.on(lane_id):
                                        delta_m = wi * warp_group_m + mi * mma_m + i
                                        delta_n = wj * warp_group_n + mj * mma_n + j
                                        in_bound = (offset_m + delta_m < m_size) and (offset_n + delta_n < n_size)
                                        if in_bound:
                                            if k_round == 0:
                                                smem_c[delta_m, delta_n] = regs_c[mi, mj, p]
                                            else:
                                                smem_c[delta_m, delta_n] += regs_c[mi, mj, p]
                                        p += 1
                        if warp_group_count_k > 1:
                            syncthreads()
                    for i, j in store_smem_c_map.on(threadIdx.x):
                        if offset_m + i < m_size and offset_n + j < n_size:
                            gmem_c[i, j] = float16(smem_c[i, j])

        ir_module = module.ir_module()
        assert isinstance(matmul_f16_kernel, Function)

        return ir_module


class MatmulF16SM90Op(Operator):
    def __init__(self, a: Tensor, b: Tensor, is_a_shared: bool, acc_dtype: Union[DataType, str], parallel_k_parts=1):
        if not (isinstance(parallel_k_parts, int) and not isinstance(parallel_k_parts, bool)):
            raise ValueError('parallel_k_parts must be an integer, got {}'.format(parallel_k_parts))
        acc_dtype = data_type(acc_dtype)
        super().__init__(
            inputs=[a, b],
            attributes={'acc_dtype': acc_dtype, 'parallel_k_parts': parallel_k_parts, 'is_a_shared': is_a_shared},
            task=MatmulF16SM90Task(input_like(a, 'a'), input_like(b, 'b'), is_a_shared, acc_dtype, parallel_k_parts),
        )


def matmul_f16_sm90(
    a: Tensor, b: Tensor, is_a_shared: bool = False, parallel_k_parts=1, acc_dtype: Union[DataType, str] = "float32"
) -> Tensor:
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError('a and b must have at least 2 dimensions, got shape {} and {}'.format(a.shape, b.shape))
    # TODO: impliment dynamic run-time shape assertion
    if not (isinstance(a.shape[-1], Expr) or isinstance(b.shape[-1], Expr)) and (
        a.shape[-1] % 2 != 0 or b.shape[-1] % 2 != 0
    ):
        raise ValueError('Expect the last dimension of the input tensors to be a multiple of 2')
    if a.dtype != dtypes.float16 or b.dtype != dtypes.float16:
        raise ValueError('BatchMatmulF16Op only support float16, got {} and {}'.format(a.dtype, b.dtype))
    acc_dtype = data_type(acc_dtype)
    return MatmulF16SM90Op(a, b, is_a_shared, acc_dtype, parallel_k_parts).outputs[0]
