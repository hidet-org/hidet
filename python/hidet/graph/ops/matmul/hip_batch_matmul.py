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
from typing import List, Tuple
import hidet
from hidet.ir import IRModule
from hidet.ir.compute import reduce
from hidet.ir.expr import is_constant, cast
from hidet.ir.layout import DataLayout, StridesLayout, row_major, column_major, local_layout
from hidet.ir.mapping import TaskMapping
from hidet.ir.type import TensorType
from hidet.lang import (
    float16,
    float32,
    i32,
    boolean,
    spatial,
    repeat,
    register_tensor,
    shared_tensor,
    attrs,
    grid,
    tensor_pointer,
)
from hidet.lang.cuda import blockIdx, threadIdx, syncthreads
from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode, compute
from hidet.graph.ops.utils import input_like
from hidet.ir.library import tune
from hidet.ir.primitives.hip.mfma import MfmaConfig, mfma_sync
from hidet.ir.primitives.hip.lds_sync import lds_sync
from hidet.ir.dtypes.vector import vectorize
from hidet.ir.primitives.hip.buffer_addr import hip_buffer_load


class BatchMatmulHipTask(Task):
    def __init__(self, a: TensorNode, b: TensorNode, mma: str = 'simt'):
        batch_size, m_size, k_size = a.shape
        batch_size, k_size, n_size = b.shape
        self.batch_size = batch_size
        self.m_size = m_size
        self.k_size = k_size
        self.n_size = n_size
        self.mma: str = mma
        c = compute(
            name='c',
            shape=[batch_size, m_size, n_size],
            fcompute=lambda r, i, j: reduce(
                shape=[k_size], fcompute=lambda k: a[r, i, k] * b[r, k, j], reduce_type='sum'
            ),
        )
        super().__init__(
            name='batch_matmul_hip',
            inputs=[a, b],
            outputs=[c],
            attributes={'batch_size': batch_size, 'm_size': m_size, 'n_size': n_size, 'k_size': k_size, 'mma': mma},
        )

    # pylint: disable=inconsistent-return-statements
    def implement_hip(self, working_dir: str) -> List[IRModule]:
        if self.inputs[0].type.dtype.nbytes > 4:
            raise ValueError('Only support data type <= 4 bytes for now')
        if self.mma == 'simt':
            capability = hidet.hip.capability()
            if capability.warpSize in (32, 64):
                if capability.warpSize == 32:
                    warp_mid = [
                        spatial(4, 8),
                        spatial(2, 16),
                        spatial(16, 2),
                        spatial(1, 32),
                        spatial(32, 1),
                        spatial(2, 1) * spatial(1, 8) * spatial(2, 1),
                    ]
                else:
                    warp_mid = [
                        spatial(8, 8),
                        spatial(4, 16),
                        spatial(16, 4),
                        spatial(2, 32),
                        spatial(32, 2),
                        spatial(4, 1) * spatial(1, 8) * spatial(2, 1),
                        spatial(2, 1) * spatial(1, 8) * spatial(4, 1),
                    ]

                @tune.space(
                    2,
                    block_warps_k=[4, 8],
                    block_warps=[(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2)],
                    warp_outer=[(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)],
                    warp_mid=warp_mid,
                    warp_inner=[(4, 4)],
                )
                @tune.space(
                    1,
                    block_warps_k=[8],
                    block_warps=[(1, 1), (1, 2), (2, 2), (2, 4)],
                    warp_outer=[(1, 1), (1, 2), (2, 1), (2, 2)],
                    warp_mid=[spatial(4, 8)] if capability.warpSize == 32 else [spatial(8, 8)],
                    warp_inner=[(4, 4), (4, 8), (8, 4)],
                )
                def schedule(
                    block_warps_k=8,
                    block_warps=(4, 2),
                    warp_outer=(2, 2),
                    warp_mid=spatial(4, 8) if capability.warpSize == 32 else spatial(8, 8),
                    warp_inner=(4, 4),
                ) -> IRModule:
                    return self.schedule_simt_amd(block_warps_k, block_warps, warp_outer, warp_mid, warp_inner)

                return tune.extract_ir_modules(schedule)
            else:
                raise ValueError('Only support warp size 32, 64 for now, got {}'.format(capability.warpSize))

        elif self.mma.startswith('mma') and hidet.hip.capability().gcnArchName == 'gfx90a':
            if self.inputs[0].type.dtype == float32:
                if hidet.option.get_search_space() == 0:
                    smem_a_layout = row_major(2, 1) * row_major(32, 32).swizzle(1).reshape([64, 16])
                    smem_b_layout = row_major(4, 1) * row_major(4, 4).swizzle(1) * row_major(1, 16)
                    return self.schedule_mma_f32_amd_gfx90a(
                        MfmaConfig.v_mfma_f32_16x16x4f32(),
                        smem_a_layout,
                        smem_b_layout,
                        spatial(2, 2),
                        vec_load_global=1,
                    )
                # TODO: minimize search space for option 1
                else:
                    irms = []
                    inst = MfmaConfig.v_mfma_f32_16x16x4f32()
                    configs = BatchMatmulHipTask.mfma_f32_generate_configs(inst.m, inst.n, inst.k)

                    for config in configs:
                        try:
                            irm = self.schedule_mma_f32_amd_gfx90a(*config)
                        except AssertionError:
                            continue
                        irms.append(irm)

                    return irms
            elif self.inputs[0].type.dtype == float16:
                if hidet.option.get_search_space() == 0:
                    ir_module = self.schedule_mma_f16_amd_gfx90a(128, 64, 64, spatial(2, 2), 2, 2, 2, True, True)
                    return [ir_module]
                else:
                    arg_names = [
                        'block_m',
                        'block_k',
                        'block_n',
                        'warp_outer',
                        'vec_load_global',
                        'vec_store_shared_a',
                        'vec_store_shared_b',
                        'swizzle_in_b',
                        'use_buffer_addr',
                    ]

                    irms = []
                    configs = BatchMatmulHipTask.mfma_f16_generate_configs()
                    for config in configs:
                        try:
                            irm = self.schedule_mma_f16_amd_gfx90a(*config)
                            tuning_kwargs = {k: v for k, v in zip(arg_names, config)}
                            setattr(irm, '_tuning_kwargs', tuning_kwargs)
                        except AssertionError:
                            continue
                        irms.append(irm)
                    return irms
            else:
                raise ValueError('Only support float16, float32 for now, got {}'.format(self.inputs[0].type.dtype))

        elif self.mma.startswith('mma'):
            raise RuntimeError(
                'mma is only supported on gfx90a for now, got {}'.format(hidet.hip.capability().gcnArchName)
            )

    def allow_prologue(self) -> bool:
        # use vectorized loading for mma, thus no prologue
        if self.mma.startswith('mma') and hidet.hip.capability().gcnArchName == 'gfx90a':
            return False
        else:
            return True

    def schedule_simt_amd(
        self,
        block_warps_k: int,
        block_warps: Tuple[int, int],
        warp_outer: Tuple[int, int],
        warp_mid: TaskMapping,
        warp_inner: Tuple[int, int],
    ) -> IRModule:
        task = self
        dtype = task.inputs[0].type.dtype
        warp_k = 1

        # Task Layouts
        warp_inner_layout = repeat(*warp_inner)
        warp_mid_layout = warp_mid
        warp_outer_layout = repeat(*warp_outer)
        warp_layout = warp_outer_layout * warp_mid_layout * warp_inner_layout
        block_warps_layout = spatial(*block_warps)
        block_layout = block_warps_layout * warp_layout
        block_k = block_warps_k * warp_k
        warp_mid_shape = warp_mid_layout.task_shape
        block_shape = block_layout.task_shape
        warp_size = warp_mid.num_workers
        block_size = block_layout.num_workers

        lines = block_size // block_k
        a_g2s_layout = spatial(lines, block_k) * repeat(block_shape[0] // lines, 1)
        b_g2s_layout = repeat(1, block_shape[1] // lines) * spatial(block_k, lines)
        a_s2r_layout = (
            block_warps_layout * repeat(warp_outer[0], warp_k) * warp_mid_layout * repeat(warp_inner[0], warp_k)
        )
        b_s2r_layout = (
            block_warps_layout * repeat(warp_k, warp_outer[1]) * warp_mid_layout * repeat(warp_k, warp_inner[1])
        )

        # Data Layouts
        regs_a_layout = (
            local_layout(block_warps[0], 1)
            * column_major(warp_outer[0], warp_k)
            * local_layout(warp_mid_shape[0], 1)
            * row_major(warp_inner[0], 1)
        )
        regs_a_layout = [2] + regs_a_layout
        regs_b_layout = (
            local_layout(1, block_warps[1])
            * row_major(warp_k, warp_outer[1])
            * local_layout(1, warp_mid_shape[1])
            * row_major(1, warp_inner[1])
        )
        regs_b_layout = [2] + regs_b_layout
        regs_c_layout = (
            local_layout(*block_warps) * row_major(*warp_outer) * local_layout(*warp_mid_shape) * row_major(*warp_inner)
        )
        regs_a_ldg_layout = local_layout(block_size // block_k, block_k) * row_major(
            block_shape[0] // (block_size // block_k), 1
        )
        regs_b_ldg_layout = row_major(1, block_shape[1] // (block_size // block_k)) * local_layout(
            block_k, block_size // block_k
        )

        used_smem_bytes_per_block = (block_shape[0] + block_shape[1]) * block_k * 2 * dtype.nbytes

        bs = task.attrs['batch_size']
        m_size = task.attrs['m_size']
        n_size = task.attrs['n_size']
        k_size = task.attrs['k_size']
        block_m, block_n = block_shape[0], block_shape[1]
        m_tiles = (m_size + block_m - 1) // block_m
        n_tiles = (n_size + block_n - 1) // block_n
        k_tiles = (k_size + block_k - 1) // block_k

        capability = hidet.hip.capability()

        reserved_regs = 48  # number of reserved registers for intermediate results
        used_num_regs_per_thread = (
            regs_a_layout.size // 2  # account for double buffering
            + regs_b_layout.size // 2
            + regs_c_layout.size
            + regs_a_ldg_layout.size
            + regs_b_ldg_layout.size
            + reserved_regs
        )
        used_num_regs_per_thread = (used_num_regs_per_thread + 7) // 8 * 8

        tune.check(warp_size == capability.warpSize)
        tune.check(block_warps_k % 2 == 0)
        tune.check(block_k <= warp_size)
        tune.check(warp_size % block_k == 0)
        tune.check(block_shape[0] % (block_size // block_k) == 0 and block_shape[1] % (block_size // block_k) == 0)
        tune.check(used_smem_bytes_per_block <= capability.sharedMemPerBlock)
        tune.check(used_num_regs_per_thread * block_size <= capability.regsPerBlock)

        with hidet.script_module() as module:

            @hidet.script
            def copy_a_g2r(
                a: dtype[bs, m_size, k_size],
                regs_a_ldg: TensorType(dtype=dtype, layout=regs_a_ldg_layout),
                offset_m: i32,
                offset_k: i32,
                first_k_tile_size: i32,
            ):
                attrs.func_kind = 'hip_internal'
                gmem_a = a[blockIdx.y, offset_m:, offset_k:]
                for i, k in a_g2s_layout.on(threadIdx.x):
                    k_predicate = (first_k_tile_size == 0) or k < first_k_tile_size
                    if offset_m + i < m_size and k_predicate:
                        regs_a_ldg[i, k] = gmem_a.read([i, k], protected=False)
                    else:
                        regs_a_ldg[i, k] = 0.0

            @hidet.script
            def copy_a_r2s(
                regs_a_ldg: TensorType(dtype=dtype, layout=regs_a_ldg_layout),
                smem_a: TensorType(dtype=dtype, layout=StridesLayout.from_shape([2, block_m, block_k], perm=[0, 2, 1])),
                buffer_idx: i32,
            ):
                attrs.func_kind = 'hip_internal'
                for i, k in a_g2s_layout.on(threadIdx.x):
                    smem_a[buffer_idx, i, k] = regs_a_ldg[i, k]

            @hidet.script
            def copy_a_s2r(
                smem_a: TensorType(dtype=dtype, layout=StridesLayout.from_shape([2, block_m, block_k], perm=[0, 2, 1])),
                regs_a: TensorType(dtype=dtype, layout=regs_a_layout),
                smem_buffer_idx: i32,
                regs_buffer_idx: i32,
                k_frag_idx: i32,
            ):
                attrs.func_kind = 'hip_internal'
                smem_a_start = smem_a[smem_buffer_idx, :, k_frag_idx:]
                for i, k in a_s2r_layout.on(threadIdx.x):
                    regs_a[regs_buffer_idx, i, k] = smem_a_start[i, 0]

            @hidet.script
            def copy_b_g2r(
                b: dtype[bs, k_size, n_size],
                regs_b_ldg: TensorType(dtype=dtype, layout=regs_b_ldg_layout),
                offset_k: i32,
                offset_n: i32,
                first_k_tile_size: i32,
            ):
                attrs.func_kind = 'hip_internal'
                gmem_b = b[blockIdx.y, offset_k:, offset_n:]
                for k, j in b_g2s_layout.on(threadIdx.x):
                    k_predicate = (first_k_tile_size == 0) or k < first_k_tile_size
                    if offset_n + j < n_size and k_predicate:
                        regs_b_ldg[k, j] = gmem_b.read([k, j], protected=False)
                    else:
                        regs_b_ldg[k, j] = 0.0

            @hidet.script
            def copy_b_r2s(
                regs_b_ldg: TensorType(dtype=dtype, layout=regs_b_ldg_layout),
                smem_b: TensorType(dtype=dtype, layout=StridesLayout.from_shape([2, block_k, block_n], perm=[0, 1, 2])),
                buffer_idx: i32,
            ):
                attrs.func_kind = 'hip_internal'
                for k, j in b_g2s_layout.on(threadIdx.x):
                    smem_b[buffer_idx, k, j] = regs_b_ldg[k, j]

            @hidet.script
            def copy_b_s2r(
                smem_b: TensorType(dtype=dtype, layout=StridesLayout.from_shape([2, block_k, block_n], perm=[0, 1, 2])),
                regs_b: TensorType(dtype=dtype, layout=regs_b_layout),
                smem_buffer_idx: i32,
                regs_buffer_idx: i32,
                k_frag_idx: i32,
            ):
                attrs.func_kind = 'hip_internal'
                smem_b_start = smem_b[smem_buffer_idx, k_frag_idx:, :]
                for k, j in b_s2r_layout.on(threadIdx.x):
                    regs_b[regs_buffer_idx, k, j] = smem_b_start[0, j]

            @hidet.script
            def copy_c_r2g(
                regs_c: TensorType(dtype=dtype, layout=regs_c_layout),
                c: dtype[bs, m_size, n_size],
                offset_m: i32,
                offset_n: i32,
            ):
                attrs.func_kind = 'hip_internal'

                gmem_c = c[blockIdx.y, offset_m:, offset_n:]
                for i, j in block_layout.on(threadIdx.x):
                    if offset_m + i < m_size and offset_n + j < n_size:
                        gmem_c.write([i, j], regs_c[i, j], protected=False)

            @hidet.script
            def mma(
                regs_a: TensorType(dtype=dtype, layout=regs_a_layout),
                regs_b: TensorType(dtype=dtype, layout=regs_b_layout),
                regs_c: TensorType(dtype=dtype, layout=regs_c_layout),
                buffer_idx: i32,
            ):
                attrs.func_kind = 'hip_internal'

                for i, j in block_layout.on(threadIdx.x):
                    for k in range(warp_k):
                        regs_c[i, j] += regs_a[buffer_idx, i, k] * regs_b[buffer_idx, k, j]

            @hidet.script
            def batch_matmul_simt_kernel(
                a: dtype[bs, m_size, k_size], b: dtype[bs, k_size, n_size], c: dtype[bs, m_size, n_size]
            ):
                attrs.func_kind = 'hip_kernel'

                attrs.hip.grid_dim = (m_tiles * n_tiles, bs)
                attrs.hip.block_dim = block_size
                # attrs.hip.dynamic_smem_bytes = cuda_dynamic_smem_bytes
                # attrs.cuda.min_blocks = min_thread_blocks

                offset_m, offset_n = blockIdx.x // n_tiles * block_m, blockIdx.x % n_tiles * block_n

                smem = shared_tensor('int8', shape=[used_smem_bytes_per_block])

                smem_a = tensor_pointer(
                    dtype,
                    layout=StridesLayout.from_shape([2, block_m, block_k], perm=[0, 2, 1]),
                    init=cast(~smem[0], ~dtype),
                )
                smem_a_bytes = smem_a.type.tensor_type.storage_bytes()
                smem_b = tensor_pointer(
                    dtype,
                    layout=StridesLayout.from_shape([2, block_k, block_n], perm=[0, 1, 2]),
                    init=cast(~smem[smem_a_bytes], ~dtype),
                )

                regs_a = register_tensor(dtype, layout=regs_a_layout)
                regs_b = register_tensor(dtype, layout=regs_b_layout)
                regs_c = register_tensor(dtype, layout=regs_c_layout)
                regs_a_ldg = register_tensor(dtype, layout=regs_a_ldg_layout)
                regs_b_ldg = register_tensor(dtype, layout=regs_b_ldg_layout)

                # Copy first k-tile from global to shared
                first_k_tile_size = k_size - (k_tiles - 1) * block_k
                copy_a_g2r(a, regs_a_ldg, offset_m, 0, first_k_tile_size)
                copy_a_r2s(regs_a_ldg, smem_a, 0)
                copy_b_g2r(b, regs_b_ldg, 0, offset_n, first_k_tile_size)
                copy_b_r2s(regs_b_ldg, smem_b, 0)
                syncthreads()
                # Copy first k-frag within first k-tile from shared to local
                copy_a_s2r(smem_a, regs_a, 0, 0, 0)
                copy_b_s2r(smem_b, regs_b, 0, 0, 0)
                syncthreads()
                # Initialize regs C
                for i, j in block_layout.on(threadIdx.x):
                    regs_c[i, j] = 0.0

                # Main k-tile loop
                for k in range(k_tiles - 1):
                    offset_k = k * block_k + first_k_tile_size
                    for k_frag in grid(block_warps_k):
                        if k_frag == block_warps_k - 1:
                            # Store next AB tile from local into shared
                            copy_a_r2s(regs_a_ldg, smem_a, (k + 1) % 2)
                            copy_b_r2s(regs_b_ldg, smem_b, (k + 1) % 2)
                            syncthreads()
                            # Load next k-fragment (from next k-tile) from shared to local
                            copy_a_s2r(smem_a, regs_a, (k + 1) % 2, (k_frag + 1) % 2, 0)
                            copy_b_s2r(smem_b, regs_b, (k + 1) % 2, (k_frag + 1) % 2, 0)
                        else:
                            # Load next k-fragment from shared to local
                            copy_a_s2r(smem_a, regs_a, k % 2, (k_frag + 1) % 2, k_frag + 1)
                            copy_b_s2r(smem_b, regs_b, k % 2, (k_frag + 1) % 2, k_frag + 1)
                        if k_frag == 0:
                            # Load next AB tile from global into local
                            copy_a_g2r(a, regs_a_ldg, offset_m, offset_k, 0)
                            copy_b_g2r(b, regs_b_ldg, offset_k, offset_n, 0)
                        # Perform MMA
                        mma(regs_a, regs_b, regs_c, k_frag % 2)
                # Perform MMA for last k-tile
                last_k = k_tiles - 1
                for k_frag in grid(block_warps_k):
                    if k_frag < block_warps_k - 1:
                        copy_a_s2r(smem_a, regs_a, last_k % 2, (k_frag + 1) % 2, k_frag + 1)
                        copy_b_s2r(smem_b, regs_b, last_k % 2, (k_frag + 1) % 2, k_frag + 1)
                    mma(regs_a, regs_b, regs_c, k_frag % 2)

                # Store results from regs_c into C
                copy_c_r2g(regs_c, c, offset_m, offset_n)

        ir_module = module.ir_module()
        return ir_module

    @staticmethod
    def mfma_f32_generate_configs(mma_m, mma_n, mma_k):
        for use_buffer_addr in [True, False]:
            for block_m in [128, 64]:
                for block_n in [128, 64]:
                    for block_k in [32, 16]:
                        for warp_map in [spatial(2, 2), spatial(2, 1), spatial(1, 2), spatial(1, 1)]:
                            for vec_load_global in [1, 2, 4]:
                                assert 1024 // block_k <= block_m

                                smem_a_layout = row_major(block_m // (1024 // block_k), 1) * row_major(32, 32).swizzle(
                                    1
                                ).reshape([1024 // block_k, block_k])
                                mid_swizzle = block_n // mma_n
                                assert block_k % mid_swizzle == 0
                                smem_b_layout = (
                                    row_major(block_k // mid_swizzle, 1)
                                    * row_major(mid_swizzle, mid_swizzle).swizzle(1)
                                    * row_major(1, mma_n)
                                )

                                # pylint: disable=line-too-long
                                yield MfmaConfig.v_mfma_f32_16x16x4f32(), smem_a_layout, smem_b_layout, warp_map, vec_load_global, use_buffer_addr

    def schedule_mma_f32_amd_gfx90a(
        self,
        inst: MfmaConfig,
        smem_a_layout: DataLayout,
        smem_b_layout: DataLayout,
        warp_outer_mapping: TaskMapping,
        vec_load_global: int = 1,
        use_buffer_addr: bool = False,
    ) -> IRModule:
        task = self
        assert len(smem_a_layout.shape) == 2
        assert len(smem_b_layout.shape) == 2
        assert smem_a_layout.shape[1] == smem_b_layout.shape[0]

        warp_m, warp_n = warp_outer_mapping.task_shape

        block_m, block_k = smem_a_layout.shape
        _, block_n = smem_b_layout.shape

        warp_size = 64
        warps = warp_m * warp_n
        threads = warps * warp_size

        used_smem_bytes_per_block = (block_m + block_n) * block_k * task.inputs[0].type.dtype.nbytes

        bs = task.attrs['batch_size']
        m_size = task.attrs['m_size']
        n_size = task.attrs['n_size']
        k_size = task.attrs['k_size']

        tiles_m = (m_size + block_m - 1) // block_m
        tiles_n = (n_size + block_n - 1) // block_n
        tiles_k = (k_size + block_k - 1) // block_k

        dtype = task.inputs[0].type.dtype
        capability = hidet.hip.capability()

        tune.check(warp_size == capability.warpSize)
        tune.check(used_smem_bytes_per_block <= capability.sharedMemPerBlock)
        # tune.check(used_num_regs_per_thread * block_size <= hidet.hip.capability().regsPerBlock)

        vectype = vectorize(dtype, vec_load_global)

        with hidet.script_module() as module:
            assert block_k % vec_load_global == 0 and block_n % vec_load_global == 0
            assert threads % (block_k // vec_load_global) == 0 and threads % (block_n // vec_load_global) == 0

            lines_a = threads // (block_k // vec_load_global)
            a_g2r_mapping = spatial(lines_a, block_k // vec_load_global) * repeat(max(1, block_m // lines_a), 1)
            regs_a_ldg_layout = local_layout(lines_a, block_k // vec_load_global) * row_major(
                max(1, block_m // lines_a), vec_load_global
            )

            @hidet.script
            def vec_read(dst: ~vectype, src: ~vectype, inbounds: boolean):
                attrs.func_kind = 'hip_internal'
                if inbounds:
                    cast(dst, ~vectype)[0] = cast(src, ~vectype)[0]
                else:
                    cast(dst, ~vectype)[0] = vectype.zero

            @hidet.script
            def copy_a_g2r(
                a: dtype[bs, m_size, k_size],
                regs_a_ldg: TensorType(dtype=dtype, layout=regs_a_ldg_layout),
                offset_m: i32,
                offset_k: i32,
            ):
                attrs.func_kind = 'hip_internal'
                if not use_buffer_addr:
                    gmem_a = a[blockIdx.y, offset_m:, offset_k:]
                    for i, k in a_g2r_mapping.on(threadIdx.x):
                        inbounds = offset_m + i < m_size and offset_k + k < k_size
                        vec_read(
                            cast(~regs_a_ldg[i, k * vec_load_global], ~vectype),
                            cast(~gmem_a[i, k * vec_load_global], ~vectype),
                            inbounds,
                        )
                else:
                    a_addr = ~a[blockIdx.y, 0, 0]
                    for i, k in a_g2r_mapping.on(threadIdx.x):
                        inbounds = offset_m + i < m_size and offset_k + k < k_size
                        lane_offset = (offset_m + i) * k_size + offset_k + k * vec_load_global
                        hip_buffer_load(
                            wave_ptr=a_addr,
                            elem_space=m_size * k_size,
                            lane_offset=lane_offset,
                            dst_ptr=~regs_a_ldg[i, k * vec_load_global],
                            dtype="float32",
                            vec_load=vec_load_global,
                        )
                        if not inbounds:
                            for vl in range(vec_load_global):
                                regs_a_ldg[i, k * vec_load_global + vl] = 0.0

            @hidet.script
            def copy_a_r2s(
                regs_a_ldg: TensorType(dtype=dtype, layout=regs_a_ldg_layout),
                smem_a: TensorType(dtype=dtype, layout=smem_a_layout),
            ):
                attrs.func_kind = 'hip_internal'
                for i, k in a_g2r_mapping.on(threadIdx.x):
                    for vl in range(vec_load_global):
                        smem_a[i, k * vec_load_global + vl] = regs_a_ldg[i, k * vec_load_global + vl]

            lines_b = threads // (block_n // vec_load_global)
            b_g2r_mapping = spatial(lines_b, block_n // vec_load_global) * repeat(max(1, block_k // lines_b), 1)
            regs_b_ldg_layout = local_layout(lines_b, block_n // vec_load_global) * row_major(
                max(1, block_k // lines_b), vec_load_global
            )

            @hidet.script
            def copy_b_g2r(
                b: dtype[bs, k_size, n_size],
                regs_b_ldg: TensorType(dtype=dtype, layout=regs_b_ldg_layout),
                offset_k: i32,
                offset_n: i32,
            ):
                attrs.func_kind = 'hip_internal'
                if not use_buffer_addr:
                    gmem_b = b[blockIdx.y, offset_k:, offset_n:]
                    for k, j in b_g2r_mapping.on(threadIdx.x):
                        inbounds = offset_n + j < n_size and offset_k + k < k_size
                        vec_read(
                            cast(~regs_b_ldg[k, j * vec_load_global], ~vectype),
                            cast(~gmem_b[k, j * vec_load_global], ~vectype),
                            inbounds,
                        )
                else:
                    b_addr = ~b[blockIdx.y, 0, 0]
                    for k, j in b_g2r_mapping.on(threadIdx.x):
                        inbounds = offset_n + j < n_size and offset_k + k < k_size
                        lane_offset = (offset_k + k) * n_size + offset_n + j * vec_load_global
                        hip_buffer_load(
                            wave_ptr=b_addr,
                            elem_space=k_size * n_size,
                            lane_offset=lane_offset,
                            dst_ptr=~regs_b_ldg[k, j * vec_load_global],
                            dtype="float32",
                            vec_load=vec_load_global,
                        )
                        if not inbounds:
                            for vl in range(vec_load_global):
                                regs_b_ldg[k, j * vec_load_global + vl] = 0.0

            @hidet.script
            def copy_b_r2s(
                regs_b_ldg: TensorType(dtype=dtype, layout=regs_b_ldg_layout),
                smem_b: TensorType(dtype=dtype, layout=smem_b_layout),
            ):
                attrs.func_kind = 'hip_internal'
                for k, j in b_g2r_mapping.on(threadIdx.x):
                    for vl in range(vec_load_global):
                        smem_b[k, j * vec_load_global + vl] = regs_b_ldg[k, j * vec_load_global + vl]

            assert block_m % (warp_m * inst.m) == 0 and block_n % (warp_n * inst.n) == 0
            rep_m = block_m // (warp_m * inst.m)
            block_regs_mma_a_mapping = repeat(rep_m, 1) * inst.a_load_map
            block_regs_mma_a_layout = local_layout(warp_m, 1) * row_major(rep_m, 1) * inst.regs_a_layout

            rep_n = block_n // (warp_n * inst.n)
            block_regs_mma_b_mapping = repeat(1, rep_n) * inst.b_load_map
            block_regs_mma_b_layout = local_layout(1, warp_n) * row_major(1, rep_n) * inst.regs_b_layout

            block_c_store_map = warp_outer_mapping * repeat(rep_m, rep_n) * inst.c_store_map
            block_regs_mma_c_layout = local_layout(warp_m, warp_n) * row_major(rep_m, rep_n) * inst.regs_c_layout

            # each warp handles a_tile[warp_tile_m, block_k] * b_tile[block_k, warp_tile_n]
            warp_tile_m = rep_m * inst.m
            warp_tile_n = rep_n * inst.n
            warp_tile_k = inst.k
            warp_map = warp_outer_mapping

            @hidet.script
            def copy_c_r2g(
                regs_c: TensorType(dtype=dtype, layout=block_regs_mma_c_layout),
                c: dtype[bs, m_size, n_size],
                offset_m: i32,
                offset_n: i32,
            ):
                attrs.func_kind = 'hip_internal'

                gmem_c = c[blockIdx.y, offset_m:, offset_n:]
                for i, j in block_c_store_map.on(threadIdx.x):
                    if offset_m + i < m_size and offset_n + j < n_size:
                        gmem_c.write([i, j], regs_c[i, j], protected=False)

            @hidet.script
            def block_gemm(
                smem_a: TensorType(dtype=dtype, layout=smem_a_layout),
                smem_b: TensorType(dtype=dtype, layout=smem_b_layout),
                regs_c: TensorType(dtype=dtype, layout=block_regs_mma_c_layout),
            ):
                attrs.func_kind = 'hip_internal'
                lane = threadIdx.x % warp_size
                warp = threadIdx.x // warp_size
                regs_a = register_tensor(dtype, layout=block_regs_mma_a_layout)
                regs_b = register_tensor(dtype, layout=block_regs_mma_b_layout)

                for ki in range(block_k // warp_tile_k):
                    for im, jn in warp_map.on(warp):
                        for i, k in block_regs_mma_a_mapping.on(lane):
                            regs_a[i + im * warp_tile_m, k] = smem_a[i + im * warp_tile_m, k + ki * warp_tile_k]

                    for im, jn in warp_map.on(warp):
                        for k, n in block_regs_mma_b_mapping.on(lane):
                            regs_b[k, n + jn * warp_tile_n] = smem_b[k + ki * warp_tile_k, n + jn * warp_tile_n]

                    for im, jn in (warp_map * repeat(rep_m, rep_n)).on(warp):
                        wm = im * inst.m
                        wn = jn * inst.n
                        mfma_sync(inst, ~regs_a[wm, 0], ~regs_b[0, wn], ~regs_c[wm, wn])

            @hidet.script
            def mma(
                regs_a: TensorType(dtype=dtype, layout=block_regs_mma_a_layout),
                regs_b: TensorType(dtype=dtype, layout=block_regs_mma_b_layout),
                regs_c: TensorType(dtype=dtype, layout=block_regs_mma_c_layout),
            ):
                attrs.func_kind = 'hip_internal'
                for im, jn in warp_map.on(threadIdx.x / warp_size):
                    mfma_sync(
                        inst,
                        ~regs_a[im * warp_tile_m, 0],
                        ~regs_b[0, jn * warp_tile_n],
                        ~regs_c[im * warp_tile_m, jn * warp_tile_n],
                    )

            @hidet.script
            def batch_matmul_mma_kernel(
                a: dtype[bs, m_size, k_size], b: dtype[bs, k_size, n_size], c: dtype[bs, m_size, n_size]
            ):
                attrs.func_kind = 'hip_kernel'

                attrs.hip.grid_dim = (tiles_n * tiles_m, bs)
                attrs.hip.block_dim = warps * warp_size
                # attrs.hip.dynamic_smem_bytes = cuda_dynamic_smem_bytes
                # attrs.cuda.min_blocks = min_thread_blocks

                offset_m, offset_n = (blockIdx.x // tiles_n) * block_m, (blockIdx.x % tiles_n) * block_n

                smem = shared_tensor('int8', shape=[used_smem_bytes_per_block])

                smem_a = tensor_pointer(dtype, layout=smem_a_layout, init=cast(~smem[0], ~dtype))
                smem_a_bytes = smem_a.type.tensor_type.storage_bytes()
                smem_b = tensor_pointer(dtype, layout=smem_b_layout, init=cast(~smem[smem_a_bytes], ~dtype))

                regs_c = register_tensor(dtype, layout=block_regs_mma_c_layout)
                regs_a_ldg = register_tensor(dtype, layout=regs_a_ldg_layout)
                regs_b_ldg = register_tensor(dtype, layout=regs_b_ldg_layout)

                for i, j in block_c_store_map.on(threadIdx.x):
                    regs_c[i, j] = 0.0

                copy_a_g2r(a, regs_a_ldg, offset_m, 0)
                copy_b_g2r(b, regs_b_ldg, 0, offset_n)
                copy_a_r2s(regs_a_ldg, smem_a)
                copy_b_r2s(regs_b_ldg, smem_b)
                lds_sync()

                for kb in range(tiles_k - 1):
                    offset_k = (kb + 1) * block_k
                    copy_a_g2r(a, regs_a_ldg, offset_m, offset_k)
                    lds_sync()
                    copy_b_g2r(b, regs_b_ldg, offset_k, offset_n)
                    block_gemm(smem_a, smem_b, regs_c)
                    lds_sync()

                    copy_a_r2s(regs_a_ldg, smem_a)
                    copy_b_r2s(regs_b_ldg, smem_b)
                lds_sync()
                block_gemm(smem_a, smem_b, regs_c)

                # Store results from regs_c into C
                copy_c_r2g(regs_c, c, offset_m, offset_n)

            assert isinstance(batch_matmul_mma_kernel, hidet.ir.Function)
            batch_matmul_mma_kernel.name = f'batch_matmul_mma_kernel_vec_ldg{vec_load_global}_buf_addr{use_buffer_addr}'

        ir_module = module.ir_module()
        return ir_module

    @staticmethod
    def mfma_f16_generate_configs():
        capacity = hidet.hip.capability()
        for block_m, block_n in [(128, 64), (256, 64), (64, 128), (256, 128)]:
            for block_k in [32, 64]:
                smem_use = (block_m + block_n) * block_k * 2
                if smem_use > capacity.sharedMemPerBlock:
                    continue
                for warp_map in [spatial(2, 2), spatial(4, 2)]:
                    wm, wn = warp_map.task_shape
                    if block_m % (wm * 32) != 0 or block_n % (wn * 32) != 0:
                        continue
                    for vec_load_global in [1, 2, 4, 8]:
                        for vec_store_shared_a in [1, 2, 4]:
                            if vec_store_shared_a > vec_load_global:
                                continue
                            for vec_store_shared_b in [1, 2, 4]:
                                if vec_store_shared_b > vec_load_global:
                                    continue
                                for swizzle_in_b in [True, False]:
                                    if not swizzle_in_b and vec_store_shared_b > 1:
                                        continue
                                    # pylint: disable=line-too-long
                                    yield block_m, block_k, block_n, warp_map, vec_load_global, vec_store_shared_a, vec_store_shared_b, swizzle_in_b, True

    def schedule_mma_f16_amd_gfx90a(
        self,
        block_m: int,
        block_k: int,
        block_n: int,
        warp_outer_mapping: TaskMapping,
        vec_load_global: int = 1,
        vec_store_shared_a: int = 2,
        vec_store_shared_b: int = 1,
        swizzle_in_b: bool = True,
        use_buffer_addr: bool = False,
    ) -> IRModule:
        # this is the best instruction for compute bound
        inst = MfmaConfig.v_mfma_f32_32x32x8f16()
        task = self
        warp_m, warp_n = warp_outer_mapping.task_shape

        warp_size = 64
        warps = warp_m * warp_n
        threads = warps * warp_size

        assert 4 <= block_k <= 1024
        smem_a_layout = row_major(block_m // (1024 // block_k), 1) * (
            row_major(16, 16).swizzle(1) * row_major(1, 4)  # 16 x 4 float16 = 32 banks
        ).reshape(
            [1024 // block_k, block_k]
        )  # assume block_k <= 1024 and block_k >= 4

        smem_b_layout = row_major(block_k // 2, block_n) * column_major(2, 1)

        assert vec_store_shared_a <= 4  # the max contiguity allowed by the shared layout
        assert vec_store_shared_b == 1 or swizzle_in_b  # cannot vec store as rows has stride 2

        assert task.inputs[0].type.dtype == task.inputs[1].type.dtype == float16
        used_smem_bytes_per_block = (block_m + block_n) * block_k * task.inputs[0].type.dtype.nbytes

        bs = task.attrs['batch_size']
        m_size = task.attrs['m_size']
        n_size = task.attrs['n_size']
        k_size = task.attrs['k_size']

        tiles_m = (m_size + block_m - 1) // block_m
        tiles_n = (n_size + block_n - 1) // block_n
        tiles_k = (k_size + block_k - 1) // block_k

        dtype = task.inputs[0].type.dtype
        capability = hidet.hip.capability()

        tune.check(warp_size == capability.warpSize)
        tune.check(used_smem_bytes_per_block <= capability.sharedMemPerBlock)
        # tune.check(used_num_regs_per_thread * block_size <= hidet.hip.capability().regsPerBlock)

        vectype = vectorize(dtype, vec_load_global)

        with hidet.script_module() as module:
            assert block_k % vec_load_global == 0 and block_n % vec_load_global == 0, "a"
            assert threads % (block_k // vec_load_global) == 0 and threads % (block_n // vec_load_global) == 0, "b"

            lines_a = threads // (block_k // vec_load_global)
            a_g2r_mapping = spatial(lines_a, block_k // vec_load_global) * repeat(max(1, block_m // lines_a), 1)
            regs_a_ldg_layout = local_layout(lines_a, block_k // vec_load_global) * row_major(
                max(1, block_m // lines_a), vec_load_global
            )

            @hidet.script
            def vec_read(dst: ~vectype, src: ~vectype, inbounds: boolean):
                attrs.func_kind = 'hip_internal'
                if inbounds:
                    cast(dst, ~vectype)[0] = cast(src, ~vectype)[0]
                else:
                    cast(dst, ~vectype)[0] = vectype.zero

            @hidet.script
            def copy_a_g2r(
                a: dtype[bs, m_size, k_size],
                regs_a_ldg: TensorType(dtype=dtype, layout=regs_a_ldg_layout),
                offset_m: i32,
                offset_k: i32,
            ):
                attrs.func_kind = 'hip_internal'
                if not use_buffer_addr:
                    gmem_a = a[blockIdx.y, offset_m:, offset_k:]
                    for i, k in a_g2r_mapping.on(threadIdx.x):
                        inbounds = offset_m + i < m_size and offset_k + k < k_size
                        vec_read(
                            cast(~regs_a_ldg[i, k * vec_load_global], ~vectype),
                            cast(~gmem_a[i, k * vec_load_global], ~vectype),
                            inbounds,
                        )
                else:
                    a_addr = ~a[blockIdx.y, 0, 0]
                    for i, k in a_g2r_mapping.on(threadIdx.x):
                        inbounds = offset_m + i < m_size and offset_k + k < k_size
                        lane_offset = (offset_m + i) * k_size + offset_k + k * vec_load_global
                        hip_buffer_load(
                            wave_ptr=a_addr,
                            elem_space=m_size * k_size,
                            lane_offset=lane_offset,
                            dst_ptr=~regs_a_ldg[i, k * vec_load_global],
                            dtype="float16",
                            vec_load=vec_load_global,
                        )
                        if not inbounds:
                            for vl in range(vec_load_global):
                                regs_a_ldg[i, k * vec_load_global + vl] = 0.0

            assert vec_load_global % vec_store_shared_a == 0, "c"
            vec_store_type_a = vectorize(dtype, vec_store_shared_a)

            @hidet.script
            def copy_a_r2s(
                regs_a_ldg: TensorType(dtype=dtype, layout=regs_a_ldg_layout),
                smem_a: TensorType(dtype=dtype, layout=smem_a_layout),
            ):
                attrs.func_kind = 'hip_internal'
                for i, k in a_g2r_mapping.on(threadIdx.x):
                    for vl in range(vec_load_global // vec_store_shared_a):
                        ptr_sa = ~smem_a[i, k * vec_load_global + vl * vec_store_shared_a]
                        ptr_ra = ~regs_a_ldg[i, k * vec_load_global + vl * vec_store_shared_a]
                        # smem_a[i, k * vec_load_global + vl] = regs_a_ldg[i, k * vec_load_global + vl]
                        cast(ptr_sa, ~vec_store_type_a)[0] = cast(ptr_ra, ~vec_store_type_a)[0]
                # syncthreads()
                # if threadIdx.x == 0 and blockIdx.x == 0:
                #     for i, j in repeat(block_m, block_k).on(threadIdx.x):
                #         printf("%f ", cast(smem_a[i, j], float32))
                #         if j == block_k - 1:
                #             printf("\n")

            lines_b = threads // (block_n // vec_load_global)
            if not swizzle_in_b:
                b_g2r_mapping = spatial(lines_b, block_n // vec_load_global)
                rep_k = max(1, block_k // lines_b)
                regs_b_ldg_layout = row_major(rep_k, vec_load_global)
            else:
                b_g2r_mapping = spatial(lines_b, block_n // vec_load_global)
                rep_k = max(1, block_k // (lines_b * 2)) * 2
                regs_b_ldg_layout = row_major(rep_k, vec_load_global)

            @hidet.script
            def copy_b_g2r(
                b: dtype[bs, k_size, n_size],
                regs_b_ldg: TensorType(dtype=dtype, layout=regs_b_ldg_layout),
                offset_k: i32,
                offset_n: i32,
            ):
                attrs.func_kind = 'hip_internal'
                if not use_buffer_addr:
                    gmem_b = b[blockIdx.y, offset_k:, offset_n:]
                    for k0, j in b_g2r_mapping.on(threadIdx.x):
                        for ki in range(rep_k):
                            k = k0 * rep_k + ki
                            inbounds = offset_n + j < n_size and offset_k + k < k_size
                            vec_read(
                                cast(~regs_b_ldg[ki, 0], ~vectype),
                                cast(~gmem_b[k, j * vec_load_global], ~vectype),
                                inbounds,
                            )
                else:
                    b_addr = ~b[blockIdx.y, 0, 0]
                    for k0, j in b_g2r_mapping.on(threadIdx.x):
                        for ki in range(rep_k):
                            k = k0 * rep_k + ki
                            inbounds = offset_n + j < n_size and offset_k + k < k_size
                            lane_offset = (offset_k + k) * n_size + offset_n + j * vec_load_global
                            hip_buffer_load(
                                wave_ptr=b_addr,
                                elem_space=k_size * n_size,
                                lane_offset=lane_offset,
                                dst_ptr=~regs_b_ldg[ki, 0],
                                dtype="float16",
                                vec_load=vec_load_global,
                            )
                            if not inbounds:
                                for vl in range(vec_load_global):
                                    regs_b_ldg[ki, vl] = 0.0

            vec_store_type_b = vectorize(dtype, vec_store_shared_b * 2)

            @hidet.script
            def copy_b_r2s(
                regs_b_ldg: TensorType(dtype=dtype, layout=regs_b_ldg_layout),
                smem_b: TensorType(dtype=dtype, layout=smem_b_layout),
            ):
                attrs.func_kind = 'hip_internal'
                if swizzle_in_b:
                    for k, j in b_g2r_mapping.on(threadIdx.x):
                        for k0 in range(rep_k // 2):
                            swizzled_regs = register_tensor(float32, [vec_load_global])
                            for jl in range(vec_load_global):
                                packed_regs = register_tensor(float32, [1])
                                cast(~packed_regs[0], ~float16)[0] = regs_b_ldg[k0 * 2, jl]  # [0-15] bits
                                cast(~packed_regs[0], ~float16)[1] = regs_b_ldg[k0 * 2 + 1, jl]  # [16-31] bits
                                swizzled_regs[jl] = packed_regs[0]
                            for jl in range(vec_load_global // vec_store_shared_b):
                                ptr_sb = ~smem_b[k0 * 2 + k * rep_k, j * vec_load_global + jl * vec_store_shared_b]
                                ptr_rb = ~swizzled_regs[jl * vec_store_shared_b]
                                cast(ptr_sb, ~vec_store_type_b)[0] = cast(ptr_rb, ~vec_store_type_b)[0]
                else:
                    for k0, j in b_g2r_mapping.on(threadIdx.x):
                        for k1 in range(rep_k):
                            # vec_store_shared_b == 1 in this case
                            for vl in range(vec_load_global):
                                k = k0 * rep_k + k1
                                smem_b[k, j * vec_load_global + vl] = regs_b_ldg[k1, vl]
                # syncthreads()
                # if threadIdx.x == 0 and blockIdx.x == 0:
                #     for i, j in repeat(block_k, block_n).on(threadIdx.x):
                #         printf("%f ", cast(smem_b[i, j], float32))
                #         if j == block_n - 1:
                #             printf("\n")

            rep_m = block_m // (warp_m * inst.m)
            rep_n = block_n // (warp_n * inst.n)
            # each warp handles a_tile[warp_tile_m, block_k] * b_tile[block_k, warp_tile_n]
            warp_tile_m = rep_m * inst.m
            warp_tile_n = rep_n * inst.n
            warp_tile_k = inst.k
            warp_map = warp_outer_mapping

            assert (
                block_m % (warp_m * 32) == 0 and block_n % (warp_n * 32) == 0
            ), "block_m: {}, warp_m: {}, block_n: {}, warp_n: {}".format(block_m, warp_m, block_n, warp_n)
            block_regs_mma_a_layout = row_major(rep_m, 4)

            vec_ty_sa = vectorize(dtype, 4)

            @hidet.script
            def copy_a_s2r(
                smem_a: TensorType(dtype=dtype, layout=smem_a_layout),
                regs_a: TensorType(dtype=dtype, layout=block_regs_mma_a_layout),
                offset_k: i32,
            ):
                attrs.func_kind = 'hip_internal'

                lane = threadIdx.x % warp_size
                warp = threadIdx.x // warp_size
                for wi, _ in warp_map.on(warp):
                    for ri in range(rep_m):
                        si = lane % 32 + ri * 32 + wi * warp_tile_m
                        sj = (lane // 32) * 4 + offset_k

                        ptr_sa = ~smem_a[si, sj]
                        ptr_ra = ~regs_a[ri, 0]
                        cast(ptr_ra, ~vec_ty_sa)[0] = cast(ptr_sa, ~vec_ty_sa)[0]

            block_regs_mma_b_layout = row_major(rep_n, 4)
            vec_ty_sb = vectorize(dtype, 2)

            @hidet.script
            def copy_b_s2r(
                smem_b: TensorType(dtype=dtype, layout=smem_b_layout),
                regs_b: TensorType(dtype=dtype, layout=block_regs_mma_b_layout),
                offset_k: i32,
            ):
                attrs.func_kind = 'hip_internal'

                lane = threadIdx.x % warp_size
                warp = threadIdx.x // warp_size
                for _, wj in warp_map.on(warp):
                    for ri in range(rep_n):
                        if swizzle_in_b:
                            for ki in range(2):
                                si = ki * 2 + (lane // 32) * 4 + offset_k
                                sj = lane % 32 + ri * 32 + wj * warp_tile_n
                                ptr_sb = ~smem_b[si, sj]
                                ptr_rb = ~regs_b[ri, ki * 2]
                                cast(ptr_rb, ~vec_ty_sb)[0] = cast(ptr_sb, ~vec_ty_sb)[0]
                        else:
                            for ki in range(4):
                                si = ki + (lane // 32) * 4 + offset_k
                                sj = lane % 32 + ri * 32 + wj * warp_tile_n
                                regs_b[ri, ki] = smem_b[si, sj]

            block_c_store_map = warp_outer_mapping * repeat(rep_m, rep_n) * inst.c_store_map
            block_regs_mma_c_layout = (
                local_layout(warp_m, warp_n)
                * row_major(rep_m, rep_n)
                * row_major(4, 1)
                * local_layout(2, 1)
                * row_major(4, 1)
                * local_layout(1, 32)
            )

            @hidet.script
            def copy_c_r2g(
                regs_c: TensorType(dtype='float32', layout=block_regs_mma_c_layout),
                c: dtype[bs, m_size, n_size],
                offset_m: i32,
                offset_n: i32,
            ):
                attrs.func_kind = 'hip_internal'

                gmem_c = c[blockIdx.y, offset_m:, offset_n:]
                for i, j in block_c_store_map.on(threadIdx.x):
                    if offset_m + i < m_size and offset_n + j < n_size:
                        gmem_c.write([i, j], regs_c[i, j], protected=False)

            @hidet.script
            def block_gemm(
                smem_a: TensorType(dtype=dtype, layout=smem_a_layout),
                smem_b: TensorType(dtype=dtype, layout=smem_b_layout),
                regs_c: TensorType(dtype=dtype, layout=block_regs_mma_c_layout),
            ):
                attrs.func_kind = 'hip_internal'
                warp = threadIdx.x // warp_size
                regs_a = register_tensor(dtype, layout=block_regs_mma_a_layout)
                regs_b = register_tensor(dtype, layout=block_regs_mma_b_layout)

                for ki in range(block_k // warp_tile_k):
                    copy_a_s2r(smem_a, regs_a, ki * warp_tile_k)
                    copy_b_s2r(smem_b, regs_b, ki * warp_tile_k)

                    for im, jn in (warp_map).on(warp):
                        for rm, rn in repeat(rep_m, rep_n).on(warp):
                            wm = (im * rep_m + rm) * inst.m
                            wn = (jn * rep_n + rn) * inst.n
                            mfma_sync(inst, ~regs_a[rm, 0], ~regs_b[rn, 0], ~regs_c[wm, wn])

            @hidet.script
            def batch_matmul_mma_kernel(
                a: dtype[bs, m_size, k_size], b: dtype[bs, k_size, n_size], c: dtype[bs, m_size, n_size]
            ):
                attrs.func_kind = 'hip_kernel'

                attrs.hip.grid_dim = (tiles_n * tiles_m, bs)
                attrs.hip.block_dim = warps * warp_size
                # attrs.hip.dynamic_smem_bytes = cuda_dynamic_smem_bytes
                # attrs.cuda.min_blocks = min_thread_blocks

                offset_m, offset_n = (blockIdx.x // tiles_n) * block_m, (blockIdx.x % tiles_n) * block_n

                smem = shared_tensor('int8', shape=[used_smem_bytes_per_block])

                smem_a = tensor_pointer(dtype, layout=smem_a_layout, init=cast(~smem[0], ~dtype))
                smem_a_bytes = smem_a.type.tensor_type.storage_bytes()
                smem_b = tensor_pointer(dtype, layout=smem_b_layout, init=cast(~smem[smem_a_bytes], ~dtype))

                regs_c = register_tensor('float32', layout=block_regs_mma_c_layout)
                regs_a_ldg = register_tensor(dtype, layout=regs_a_ldg_layout)
                regs_b_ldg = register_tensor(dtype, layout=regs_b_ldg_layout)

                for i, j in block_c_store_map.on(threadIdx.x):
                    regs_c[i, j] = 0.0

                copy_a_g2r(a, regs_a_ldg, offset_m, 0)
                copy_b_g2r(b, regs_b_ldg, 0, offset_n)
                copy_a_r2s(regs_a_ldg, smem_a)
                copy_b_r2s(regs_b_ldg, smem_b)
                lds_sync()
                syncthreads()

                for kb in range(tiles_k - 1):
                    offset_k = (kb + 1) * block_k
                    copy_a_g2r(a, regs_a_ldg, offset_m, offset_k)
                    lds_sync()
                    copy_b_g2r(b, regs_b_ldg, offset_k, offset_n)

                    block_gemm(smem_a, smem_b, regs_c)
                    lds_sync()

                    copy_a_r2s(regs_a_ldg, smem_a)
                    copy_b_r2s(regs_b_ldg, smem_b)
                lds_sync()
                block_gemm(smem_a, smem_b, regs_c)

                # Store results from regs_c into C
                copy_c_r2g(regs_c, c, offset_m, offset_n)

            assert isinstance(batch_matmul_mma_kernel, hidet.ir.Function)
            batch_matmul_mma_kernel.name = (
                f'batch_matmul_mma_kernel_vec_ldg{vec_load_global}_buf_addr{use_buffer_addr}_bswizzle{swizzle_in_b}'
            )

        ir_module = module.ir_module()
        return ir_module


class BatchMatmulHipOp(Operator):
    def __init__(self, a: Tensor, b: Tensor, mma: str = 'simt'):
        # if is_false(a.shape[0] == b.shape[0]) or is_false(a.shape[2] == b.shape[1]):
        #     raise
        if not (
            len(a.shape) == len(b.shape) == 3
            and (not is_constant(a.shape[0], b.shape[0]) or a.shape[0] == b.shape[0])
            and (not is_constant(a.shape[2], b.shape[1]) or a.shape[2] == b.shape[1])
        ):
            raise ValueError(
                'Matrix multiplication expect tensor A and B with shape [B, M, K] and [B, K, N]'
                + ', got {} and {}'.format(a.shape, b.shape)
            )
        task = BatchMatmulHipTask(input_like(a, 'a'), input_like(b, 'b'), mma)
        super().__init__(inputs=[a, b], attributes={'mma': mma}, task=task)


def hip_batch_matmul(a: Tensor, b: Tensor, mma: str = 'simt') -> Tensor:
    """Batched matrix multiplication.

    Parameters
    ----------
    a: Tensor
        The lhs operand with shape [batch_size, m_size, k_size].

    b: Tensor
        The rhs operand with shape [batch_size, k_size, n_size].

    mma: str
        The matrix-multiplication-accumulate (mma) in warp level:

        - 'simt':
           Use cuda core to do the warp-level mma (simt stands for single-instruction-multiple-threads).
        - 'mma':
           Use mma instruction.
    Returns
    -------
    c: Tensor
        The result tensor of matrix multiplication.
    """
    return BatchMatmulHipOp(a, b, mma).outputs[0]
