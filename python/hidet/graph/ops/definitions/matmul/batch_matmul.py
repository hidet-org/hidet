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
from typing import List
import hidet
from hidet.ir import IRModule
from hidet.ir.compute import reduce
from hidet.ir.layout import DataLayout, StridesLayout, data_layout
from hidet.ir.mapping import TaskMapping
from hidet.ir.type import data_type, TensorType, DataType
from hidet.lang import i32, spatial, repeat, tensor, attr, grid, tensor_pointer
from hidet.lang.cuda import blockIdx, threadIdx, syncthreads
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, compute
from hidet.graph.ops.definitions.utils import input_like, tune, schedule_utils
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_sync


class BatchMatmulTask(Task):
    def __init__(self, a: TensorNode, b: TensorNode, mma: str = 'simt'):
        batch_size, m_size, k_size = a.const_shape()
        batch_size, k_size, n_size = b.const_shape()
        self.batch_size: int = batch_size
        self.m_size: int = m_size
        self.k_size: int = k_size
        self.n_size: int = n_size
        self.mma: str = mma
        c = compute(
            name='c',
            shape=[batch_size, m_size, n_size],
            fcompute=lambda r, i, j: reduce(
                shape=[k_size], fcompute=lambda k: a[r, i, k] * b[r, k, j], reduce_type='sum'
            ),
        )
        super().__init__(
            name='batch_matmul',
            inputs=[a, b],
            outputs=[c],
            attributes={'batch_size': batch_size, 'm_size': m_size, 'n_size': n_size, 'k_size': k_size, 'mma': mma},
        )

    def implement_cuda(self, working_dir: str) -> List[IRModule]:
        if self.inputs[0].type.dtype.nbytes > 4:
            raise ValueError('Only support data type <= 4 bytes for now')
        if self.mma == 'simt':
            return tune.extract_ir_modules(self.schedule_simt)
        elif self.mma.startswith('mma'):
            return tune.extract_ir_modules(self.schedule_mma)
        else:
            raise ValueError('Can not recognize mma type {}, candidates: {}'.format(self.mma, ['simt', 'mma']))

    @tune.space(2, 'block_warps_k', [4, 8])
    @tune.space(2, 'block_warps', [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2)])
    @tune.space(2, 'warp_outer', [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)])
    @tune.space(
        2,
        'warp_mid',
        [
            spatial(4, 8),
            spatial(2, 16),
            spatial(16, 2),
            spatial(1, 32),
            spatial(32, 1),
            spatial(2, 1) * spatial(1, 8) * spatial(2, 1),
        ],
    )
    @tune.space(2, 'warp_inner', [(4, 4)])
    @tune.space(1, 'block_warps_k', [8])
    @tune.space(1, 'block_warps', [(1, 1), (1, 2), (2, 2), (2, 4)])
    @tune.space(1, 'warp_outer', [(1, 1), (1, 2), (2, 1), (2, 2)])
    @tune.space(1, 'warp_mid', [spatial(4, 8)])
    @tune.space(1, 'warp_inner', [(4, 4), (4, 8), (8, 4)])
    def schedule_simt(
        self,
        block_warps_k=8,
        block_warps=(4, 2),
        warp_outer=(2, 2),
        atom_layout=TaskMapping.row_major([4, 8]),
        warp_mid=spatial(4, 8),
        warp_inner=(4, 4),
    ) -> IRModule:
        task = self
        local_layout = DataLayout.local
        row_major = DataLayout.row_major
        col_major = DataLayout.column_major
        dtype = task.inputs[0].type.dtype
        warp_k = 1

        # Task Layouts
        warp_inner_layout = repeat(*warp_inner, attrs='u+u+')
        warp_mid_layout = warp_mid
        warp_outer_layout = repeat(*warp_outer, attrs='u+u+')
        warp_layout = warp_outer_layout * warp_mid_layout * warp_inner_layout
        block_warps_layout = spatial(*block_warps)
        block_layout = block_warps_layout * warp_layout
        block_k = block_warps_k * warp_k
        warp_mid_shape = warp_mid_layout.task_shape
        block_shape = block_layout.task_shape
        warp_size = 32
        block_size = block_layout.num_workers

        lines = block_size // block_k
        a_g2s_layout = spatial(lines, block_k) * repeat(block_shape[0] // lines, 1, attrs='u+.')
        b_g2s_layout = repeat(1, block_shape[1] // lines, attrs='.u+') * spatial(block_k, lines)
        a_s2r_layout = (
            block_warps_layout
            * repeat(warp_outer[0], warp_k, attrs='u+u+')
            * warp_mid_layout
            * repeat(warp_inner[0], warp_k, attrs='u+u+')
        )
        b_s2r_layout = (
            block_warps_layout
            * repeat(warp_k, warp_outer[1], attrs='u+u+')
            * warp_mid_layout
            * repeat(warp_k, warp_inner[1], attrs='u+u+')
        )

        # Data Layouts
        regs_a_layout = (
            local_layout((block_warps[0], 1))
            * col_major((warp_outer[0], warp_k))
            * local_layout((warp_mid_shape[0], 1))
            * row_major((warp_inner[0], 1))
        )
        regs_a_layout = [2] + regs_a_layout
        regs_b_layout = (
            local_layout((1, block_warps[1]))
            * row_major((warp_k, warp_outer[1]))
            * local_layout((1, warp_mid_shape[1]))
            * row_major((1, warp_inner[1]))
        )
        regs_b_layout = [2] + regs_b_layout
        regs_c_layout = (
            local_layout(block_warps) * row_major(warp_outer) * local_layout(warp_mid_shape) * row_major(warp_inner)
        )
        regs_a_ldg_layout = local_layout((block_size // block_k, block_k)) * row_major(
            (block_shape[0] // (block_size // block_k), 1)
        )
        regs_b_ldg_layout = row_major((1, block_shape[1] // (block_size // block_k))) * local_layout(
            (block_k, block_size // block_k)
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

        reserved_regs = 48  # number of reserved registers for intermediate results
        used_num_regs_per_thread = (
            regs_a_layout.size // 2  # account for double buffering
            + regs_b_layout.size // 2
            + regs_c_layout.size
            + regs_a_ldg_layout.size
            + regs_b_ldg_layout.size
            + reserved_regs
        )
        # the number of registers allocated to each thread is a multiple of 8.
        used_num_regs_per_thread = (used_num_regs_per_thread + 7) // 8 * 8
        tune.check(used_num_regs_per_thread <= 255)
        resident_blocks = hidet.cuda.properties().regsPerMultiprocessor // (used_num_regs_per_thread * block_size)
        max_smem_bytes_per_block = (
            min(48 * 1024 // resident_blocks, hidet.cuda.properties().sharedMemPerBlock) // 128 * 128
        )

        tune.check(warp_mid.num_workers == 32)
        tune.check(block_warps_k % 2 == 0)
        tune.check(block_k <= warp_size)
        tune.check(warp_size % block_k == 0)
        tune.check(block_shape[0] % (block_size // block_k) == 0 and block_shape[1] % (block_size // block_k) == 0)
        tune.check(used_smem_bytes_per_block <= max_smem_bytes_per_block)
        tune.check(used_num_regs_per_thread * block_size <= hidet.cuda.properties().regsPerBlock)
        use_dynamic_smem = used_smem_bytes_per_block > 48 * 1024
        min_thread_blocks = resident_blocks
        cuda_dynamic_smem_bytes = used_smem_bytes_per_block if use_dynamic_smem else 0

        with hidet.script_module() as module:

            @hidet.script
            def copy_a_g2r(
                a: dtype[bs, m_size, k_size],
                regs_a_ldg: TensorType(dtype=dtype, layout=regs_a_ldg_layout),
                offset_m: i32,
                offset_k: i32,
                first_k_tile_size: i32,
            ):
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
                for i, j in block_layout.on(threadIdx.x):
                    for k in range(warp_k):
                        regs_c[i, j] += regs_a[buffer_idx, i, k] * regs_b[buffer_idx, k, j]

            @hidet.script
            def batch_matmul_kernel(
                a: dtype[bs, m_size, k_size], b: dtype[bs, k_size, n_size], c: dtype[bs, m_size, n_size]
            ):
                attr.cuda_grid_dim = (m_tiles * n_tiles, bs)
                attr.cuda_block_dim = block_size
                attr.cuda_dynamic_smem_bytes = cuda_dynamic_smem_bytes
                attr.cuda_min_blocks = min_thread_blocks

                offset_m, offset_n = blockIdx.x // n_tiles * block_m, blockIdx.x % n_tiles * block_n

                smem = tensor('shared', 'int8', shape=[used_smem_bytes_per_block])

                smem_a = tensor_pointer(dtype, layout=StridesLayout.from_shape([2, block_m, block_k], perm=[0, 2, 1]))
                smem_b = tensor_pointer(dtype, layout=StridesLayout.from_shape([2, block_k, block_n], perm=[0, 1, 2]))
                smem_a_bytes = smem_a.type.tensor_type.storage_bytes()
                smem_a = ~smem[0]
                smem_b = ~smem[smem_a_bytes]

                regs_a = tensor('register', dtype, layout=regs_a_layout)
                regs_b = tensor('register', dtype, layout=regs_b_layout)
                regs_c = tensor('register', dtype, layout=regs_c_layout)
                regs_a_ldg = tensor('register', dtype, layout=regs_a_ldg_layout)
                regs_b_ldg = tensor('register', dtype, layout=regs_b_ldg_layout)

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
                        copy_a_s2r(smem_a, regs_a, (last_k) % 2, (k_frag + 1) % 2, k_frag + 1)
                        copy_b_s2r(smem_b, regs_b, (last_k) % 2, (k_frag + 1) % 2, k_frag + 1)
                    mma(regs_a, regs_b, regs_c, (k_frag) % 2)

                # Store results from regs_c into C
                copy_c_r2g(regs_c, c, offset_m, offset_n)

        ir_module = module.ir_module()
        return ir_module

    @tune.space(2, 'block_m', [16, 32, 64, 128, 256])
    @tune.space(2, 'block_n', [8, 16, 32, 64, 128])
    @tune.space(2, 'block_k', [8, 16, 32])
    @tune.space(2, 'warp_m', [16, 32, 64])
    @tune.space(2, 'warp_n', [8, 16, 32, 64])
    @tune.space(2, 'warp_k', [8, 16, 32])
    @tune.space(2, 'mma_config', MmaConfig.all())
    @tune.space(1, 'block_m', [64, 128, 256])
    @tune.space(1, 'block_n', [64, 128])
    @tune.space(1, 'block_k', [8, 16, 32])
    @tune.space(1, 'warp_m', [32, 64])
    @tune.space(1, 'warp_n', [32, 64])
    @tune.space(1, 'warp_k', [8, 16, 32])
    @tune.space(1, 'mma_config', MmaConfig.all())
    def schedule_mma(
        self, block_m=64, block_n=64, block_k=16, warp_m=32, warp_n=32, warp_k=16, mma_config=None
    ) -> IRModule:
        def resolve_mma_type(a_dtype: DataType, b_dtype: DataType, c_dtype: DataType):
            dtype_rank = {'float16': 0, 'bfloat16': 1, 'tfloat32': 2, 'float32': 4}
            ab_rank = max(dtype_rank[a_dtype.name], dtype_rank[b_dtype.name])
            if ab_rank <= dtype_rank['float16']:
                if c_dtype == 'float32':
                    return 'mma_f16_f32'
                else:
                    return 'mma_f16_f16'
            elif ab_rank <= dtype_rank['bfloat16']:
                return 'mma_bf16_f32'
            else:
                return 'mma_tf32_f32'

        task = self
        row_major = DataLayout.row_major

        input_a, input_b, input_c = task.inputs[0], task.inputs[1], task.outputs[0]
        input_a_dtype, input_b_dtype, input_c_dtype = [t.type.dtype for t in [input_a, input_b, input_c]]
        mma_type = resolve_mma_type(input_a_dtype, input_b_dtype, input_c_dtype)

        # Resolve parameters when space level is 0
        if mma_config is None:
            default_schedule = {
                'mma_f16_f16': ([64, 128, 16], [64, 64, 16], MmaConfig.m16n8k8_f16_f16()),
                'mma_f16_f32': ([128, 64, 16], [64, 64, 16], MmaConfig.m16n8k8_f16_f32()),
                'mma_bf16_f32': ([128, 64, 16], [64, 64, 16], MmaConfig.m16n8k8_bf16_f32()),
                'mma_tf32_f32': ([64, 64, 16], [32, 32, 16], MmaConfig.m16n8k8_tf32_f32()),
            }
            block_m, block_n, block_k = default_schedule[mma_type][0]
            warp_m, warp_n, warp_k = default_schedule[mma_type][1]
            mma_config = default_schedule[mma_type][2]

        head, input_dtype, output_dtype = mma_type.split('_')  # pylint: disable=unused-variable
        tune.check(mma_config.input_dtype == input_dtype and mma_config.output_dtype == output_dtype)

        mma_m, mma_n, mma_k = (mma_config.m, mma_config.n, mma_config.k)

        tune.check(block_m % warp_m == 0 and block_n % warp_n == 0 and block_k % warp_k == 0)
        tune.check(warp_m % mma_m == 0 and warp_n % mma_n == 0 and warp_k % mma_k == 0)
        warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
        mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
        num_warps = warp_count_m * warp_count_n * warp_count_k
        block_size = num_threads = num_warps * 32
        tune.check(num_threads <= 1024)
        tune.check(block_m * block_k % num_threads == 0)
        tune.check(block_k * block_n % num_threads == 0)

        a_g2s_layout, regs_a_ldg_layout = schedule_utils.get_transfer_task_map(
            task_shape=[block_m, block_k], num_workers=num_threads, ranks=[0, 1]
        )
        b_g2s_layout, regs_b_ldg_layout = schedule_utils.get_transfer_task_map(
            task_shape=[block_k, block_n], num_workers=num_threads, ranks=[0, 1]
        )

        smem_a_layout = data_layout([2, block_m, block_k], ranks=[0, 1, 2])
        smem_b_layout = data_layout([2, block_k, block_n], ranks=[0, 1, 2])
        smem_c_layout = data_layout([block_m, block_n], ranks=[0, 1])
        regs_a_layout = row_major([2, mma_count_m, mma_config.a_elements])
        regs_b_layout = row_major([2, mma_count_n, mma_config.b_elements])
        regs_c_layout = row_major([mma_count_m, mma_count_n, mma_config.c_elements])
        smem_storage_nbytes = max(
            (smem_a_layout.size + smem_b_layout.size) * data_type(mma_config.input_dtype).nbytes,
            smem_c_layout.size * data_type(mma_config.output_dtype).nbytes,
        )
        used_registers = (
            (regs_a_layout.size + regs_b_layout.size + regs_a_ldg_layout.size + regs_b_ldg_layout.size)
            * data_type(mma_config.input_dtype).nbytes
            + regs_c_layout.size * data_type(mma_config.output_dtype).nbytes
        ) // 4 + 24
        used_registers = (used_registers + 7) // 8 * 8
        tune.check(smem_storage_nbytes <= 48 * 1024)
        tune.check(used_registers <= 255)
        tune.check(used_registers * num_threads <= hidet.cuda.properties().regsPerBlock)

        bs = task.attrs['batch_size']
        m_size = task.attrs['m_size']
        n_size = task.attrs['n_size']
        k_size = task.attrs['k_size']
        m_tiles = (m_size + block_m - 1) // block_m
        n_tiles = (n_size + block_n - 1) // block_n
        k_tiles = (k_size + block_k - 1) // block_k

        a_dtype = data_type(mma_config.input_dtype)
        b_dtype = data_type(mma_config.input_dtype)
        c_dtype = data_type(mma_config.output_dtype)
        a_zero, b_zero, c_zero = [dtype.zero for dtype in [a_dtype, b_dtype, c_dtype]]

        with hidet.script_module() as module:

            @hidet.script
            def copy_a_g2r(
                a: a_dtype[bs, m_size, k_size],
                regs_a_ldg: TensorType(dtype=a_dtype, layout=regs_a_ldg_layout),
                offset_m: i32,
                offset_k: i32,
            ):
                gmem_a = a[blockIdx.y, offset_m:, offset_k:]
                for i, k in a_g2s_layout.on(threadIdx.x):
                    if offset_m + i < m_size and offset_k + k < k_size:
                        regs_a_ldg[i, k] = gmem_a.read([i, k], protected=False)
                    else:
                        regs_a_ldg[i, k] = a_zero

            @hidet.script
            def copy_a_r2s(
                regs_a_ldg: TensorType(dtype=a_dtype, layout=regs_a_ldg_layout),
                smem_a: TensorType(dtype=a_dtype, layout=smem_a_layout),
                buffer_idx: i32,
            ):
                for i, k in a_g2s_layout.on(threadIdx.x):
                    smem_a[buffer_idx, i, k] = regs_a_ldg[i, k]

            @hidet.script
            def copy_a_s2r(
                smem_a: TensorType(dtype=a_dtype, shape=[block_m, block_k]),
                regs_a: TensorType(dtype=a_dtype, layout=regs_a_layout),
                regs_buffer_idx: i32,
            ):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, _, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    for mma_i in grid(mma_count_m, attrs='u+'):
                        p = 0
                        for i, k in mma_config.a_load_map.on(lane_id):
                            regs_a[regs_buffer_idx, mma_i, p] = smem_a[wi * warp_m + mma_i * mma_m + i, wk * warp_k + k]
                            p += 1

            @hidet.script
            def copy_b_g2r(
                b: b_dtype[bs, k_size, n_size],
                regs_b_ldg: TensorType(dtype=b_dtype, layout=regs_b_ldg_layout),
                offset_k: i32,
                offset_n: i32,
            ):
                gmem_b = b[blockIdx.y, offset_k:, offset_n:]
                for k, j in b_g2s_layout.on(threadIdx.x):
                    if offset_n + j < n_size and offset_k + k < k_size:
                        regs_b_ldg[k, j] = gmem_b.read([k, j], protected=False)
                    else:
                        regs_b_ldg[k, j] = b_zero

            @hidet.script
            def copy_b_r2s(
                regs_b_ldg: TensorType(dtype=b_dtype, layout=regs_b_ldg_layout),
                smem_b: TensorType(dtype=b_dtype, layout=smem_b_layout),
                buffer_idx: i32,
            ):
                for k, j in b_g2s_layout.on(threadIdx.x):
                    smem_b[buffer_idx, k, j] = regs_b_ldg[k, j]

            @hidet.script
            def copy_b_s2r(
                smem_b: TensorType(dtype=b_dtype, shape=[block_k, block_n]),
                regs_b: TensorType(dtype=b_dtype, layout=regs_b_layout),
                regs_buffer_idx: i32,
            ):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for _, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    for mma_j in grid(mma_count_n, attrs='u+'):
                        p = 0
                        for k, j in mma_config.b_load_map.on(lane_id):
                            regs_b[regs_buffer_idx, mma_j, p] = smem_b[wk * warp_k + k, wj * warp_n + mma_j * mma_n + j]
                            p += 1

            @hidet.script
            def copy_c_r2g(
                regs_c: TensorType(dtype=c_dtype, layout=regs_c_layout),
                c: c_dtype[bs, m_size, n_size],
                offset_m: i32,
                offset_n: i32,
            ):
                gmem_c = c[blockIdx.y, offset_m:, offset_n:]
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                if warp_count_k == 1:
                    for wi, wj in spatial(warp_count_m, warp_count_n).on(warp_id):
                        for mma_i, mma_j in grid(mma_count_m, mma_count_n, attrs='u+u+'):
                            p = 0
                            for i, j in mma_config.c_store_map.on(lane_id):
                                delta_m = wi * warp_m + mma_i * mma_m + i
                                delta_n = wj * warp_n + mma_j * mma_n + j
                                if delta_m < m_size - offset_m and delta_n < n_size - offset_n:
                                    gmem_c.write([delta_m, delta_n], regs_c[mma_i, mma_j, p])
                                p += 1
                else:
                    smem = tensor('shared', 'int8', shape=[smem_storage_nbytes])
                    smem_c = tensor_pointer(c_dtype, layout=smem_c_layout)
                    smem_c = ~smem[0]
                    for warp_k_round in grid(warp_count_k, attrs='u+'):
                        for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                            if wk == warp_k_round:
                                for mma_i, mma_j in grid(mma_count_m, mma_count_n, attrs='u+u+'):
                                    p = 0
                                    for i, j in mma_config.c_store_map.on(lane_id):
                                        delta_m = wi * warp_m + mma_i * mma_m + i
                                        delta_n = wj * warp_n + mma_j * mma_n + j
                                        if delta_m < m_size - offset_m and delta_n < n_size - offset_n:
                                            if warp_k_round == 0:
                                                smem_c[delta_m, delta_n] = regs_c[mma_i, mma_j, p]
                                            elif warp_k_round < warp_count_k - 1:
                                                smem_c[delta_m, delta_n] += regs_c[mma_i, mma_j, p]
                                            else:
                                                gmem_c.write(
                                                    [delta_m, delta_n],
                                                    smem_c[delta_m, delta_n] + regs_c[mma_i, mma_j, p],
                                                )
                                        p += 1
                        if warp_k_round + 1 != warp_count_k:
                            syncthreads()

            @hidet.script
            def mma(
                regs_a: TensorType(dtype=a_dtype, layout=regs_a_layout),
                regs_b: TensorType(dtype=b_dtype, layout=regs_b_layout),
                regs_c: TensorType(dtype=c_dtype, layout=regs_c_layout),
                buffer_idx: i32,
            ):
                for mma_i, mma_j in grid(mma_count_m, mma_count_n, attrs='u+u+'):
                    mma_sync(
                        mma_config,
                        ~regs_a[buffer_idx, mma_i, 0],
                        ~regs_b[buffer_idx, mma_j, 0],
                        ~regs_c[mma_i, mma_j, 0],
                    )

            @hidet.script
            def batch_matmul_kernel(
                a: input_a_dtype[bs, m_size, k_size],
                b: input_b_dtype[bs, k_size, n_size],
                c: input_c_dtype[bs, m_size, n_size],
            ):
                attr.cuda_grid_dim = (m_tiles * n_tiles, bs)
                attr.cuda_block_dim = block_size

                gmem_a = a[blockIdx.y, :, :]
                gmem_b = b[blockIdx.y, :, :]

                smem = tensor('shared', 'int8', shape=[smem_storage_nbytes])
                smem_a = tensor_pointer(a_dtype, layout=smem_a_layout)
                smem_b = tensor_pointer(b_dtype, layout=smem_b_layout)
                smem_a_bytes = smem_a.type.tensor_type.storage_bytes()
                smem_a = ~smem[0]
                smem_b = ~smem[smem_a_bytes]

                regs_a = tensor('register', a_dtype, layout=regs_a_layout)
                regs_b = tensor('register', b_dtype, layout=regs_b_layout)
                regs_c = tensor('register', c_dtype, layout=regs_c_layout)
                regs_a_ldg = tensor('register', a_dtype, layout=regs_a_ldg_layout)
                regs_b_ldg = tensor('register', b_dtype, layout=regs_b_ldg_layout)
                # Initialize regs C
                for i, j, p in grid(mma_count_m, mma_count_n, mma_config.c_elements):
                    regs_c[i, j, p] = c_zero

                offset_m, offset_n = blockIdx.x // n_tiles * block_m, blockIdx.x % n_tiles * block_n

                # Copy first k-tile from global to shared
                first_k_tile_size = k_size - (k_tiles - 1) * block_k
                # copy_a_g2r(a, regs_a_ldg, offset_m, 0)
                gmem_a = a[blockIdx.y, offset_m:, :]
                for i, k in a_g2s_layout.on(threadIdx.x):
                    if offset_m + i < m_size and k < first_k_tile_size:
                        regs_a_ldg[i, k] = gmem_a.read([i, k], protected=False)
                    else:
                        regs_a_ldg[i, k] = a_zero
                copy_a_r2s(regs_a_ldg, smem_a, 0)
                # copy_b_g2r(b, regs_b_ldg, 0, offset_n)
                gmem_b = b[blockIdx.y, :, offset_n:]
                for k, j in b_g2s_layout.on(threadIdx.x):
                    if offset_n + j < n_size and k < first_k_tile_size:
                        regs_b_ldg[k, j] = gmem_b.read([k, j], protected=False)
                    else:
                        regs_b_ldg[k, j] = b_zero
                copy_b_r2s(regs_b_ldg, smem_b, 0)
                syncthreads()
                # Copy first k-tile from shared to local
                copy_a_s2r(~smem_a[0, 0, 0], regs_a, 0)
                copy_b_s2r(~smem_b[0, 0, 0], regs_b, 0)

                for k0 in grid(k_tiles - 1, attrs='u2'):
                    ko = 0
                    if mma_count_k % 2 != 0 and k0 % 2 != 0:
                        ko = 1
                    for k1 in grid(mma_count_k, attrs='u+'):
                        if k1 == 0:
                            offset_k = k0 * block_k + first_k_tile_size
                            copy_a_g2r(a, regs_a_ldg, offset_m, offset_k)
                            copy_b_g2r(b, regs_b_ldg, offset_k, offset_n)
                        if k1 == mma_count_k - 1:
                            copy_a_r2s(regs_a_ldg, smem_a, (k0 + 1) % 2)
                            copy_b_r2s(regs_b_ldg, smem_b, (k0 + 1) % 2)
                            syncthreads()
                            copy_a_s2r(~smem_a[(k0 + 1) % 2, 0, 0], regs_a, (k1 + ko + 1) % 2)
                            copy_b_s2r(~smem_b[(k0 + 1) % 2, 0, 0], regs_b, (k1 + ko + 1) % 2)
                        else:
                            copy_a_s2r(~smem_a[k0 % 2, 0, (k1 + 1) * mma_k], regs_a, (k1 + ko + 1) % 2)
                            copy_b_s2r(~smem_b[k0 % 2, (k1 + 1) * mma_k, 0], regs_b, (k1 + ko + 1) % 2)
                        mma(regs_a, regs_b, regs_c, (k1 + ko) % 2)
                last_k = k_tiles - 1
                ko = 0
                if mma_count_k % 2 != 0 and last_k % 2 != 0:
                    ko = 1
                for k1 in grid(mma_count_k, attrs='u+'):
                    if k1 < mma_count_k - 1:
                        copy_a_s2r(~smem_a[last_k % 2, 0, (k1 + 1) * mma_k], regs_a, (k1 + ko + 1) % 2)
                        copy_b_s2r(~smem_b[last_k % 2, (k1 + 1) * mma_k, 0], regs_b, (k1 + ko + 1) % 2)
                    mma(regs_a, regs_b, regs_c, (k1 + ko) % 2)
                copy_c_r2g(regs_c, c, offset_m, offset_n)

        ir_module = module.ir_module()
        return ir_module


class BatchMatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor, mma: str = 'simt'):
        if not (len(a.shape) == len(b.shape) == 3 and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]):
            raise ValueError(
                'Matrix multiplication expect tensor A and B with shape [B, M, K] and [B, K, N]'
                + ', got {} and {}'.format(a.shape, b.shape)
            )
        task = BatchMatmulTask(input_like(a, 'a'), input_like(b, 'b'), mma)
        super().__init__(inputs=[a, b], attributes={'mma': mma}, task=task)


def batch_matmul(a: Tensor, b: Tensor, mma: str = 'simt') -> Tensor:
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

        See also:
        https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions

    Returns
    -------
    c: Tensor
        The result tensor of matrix multiplication.
    """
    return BatchMatmulOp(a, b, mma).get_output(0)
