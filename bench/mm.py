import hidet
import numpy as np
from typing import List, Callable, Any, Union, Optional, Dict
from hidet.ir import IRModule
from hidet.ir.compute import ReduceOperation, reduce
from hidet.ir.layout import DataLayout, StridesLayout
from hidet.ir.mapping import TaskMapping
from hidet.ir.type import data_type, TensorType, TensorPointerType
from hidet.graph.ops.definitions.matmul.batch_matmul import BatchMatmulOp
from hidet.lang import f16, f32, i32, spatial, repeat, tensor, attr, grid, printf, cast, tensor_pointer
from hidet.lang.mapping import repeat, spatial
from hidet.lang.cuda import blockIdx, threadIdx, syncthreads
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, InverseMap, compute, input_like, broadcast_shape, broadcast_shapes, broadcast_indices
from hidet.graph.ops.definitions.utils import tune



class MatMulTask(Task):
    def __init__(self, name: str, x: TensorNode, y: TensorNode):
        batch_size, m_size, k_size = x.const_shape()
        batch_size, k_size, n_size = y.const_shape()

        z = compute(
            name='z',
            shape=[batch_size, m_size, n_size],
            fcompute=lambda b, i, j: reduce(
                shape=[k_size],
                fcompute=lambda k: x[b, i, k] * y[b, k, j],
                reduce_type='sum'
            )
        )

        super().__init__(
            name=name,
            inputs=[x, y],
            outputs=[z],
            attributes={
                'batch_size': batch_size,
                'm_size': m_size,
                'n_size': n_size,
                'k_size': k_size
            }
        )

    def implement_cuda(self, workding_dir: str) -> List[IRModule]:
        return tune.extract_ir_modules(self.cuda_schedule_matmul)


    @tune.space(2, 'block_warps_k', [4, 8])
    @tune.space(2, 'block_warps', [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2)])
    @tune.space(2, 'warp_outer', [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)])
    @tune.space(2, 'warp_mid', [spatial(4, 8), spatial(2, 16), spatial(16, 2), spatial(1, 32), spatial(32, 1), spatial(2, 1) * spatial(1, 8) * spatial(2, 1)])
    @tune.space(2, 'warp_inner', [(4, 4)])
    @tune.space(1, 'block_warps_k', [8])
    @tune.space(1, 'block_warps', [(1, 1), (1, 2), (2, 2), (2, 4)])
    @tune.space(1, 'warp_outer', [(1, 1), (1, 2), (2, 1), (2, 2)])
    @tune.space(1, 'warp_mid', [spatial(4, 8)])
    @tune.space(1, 'warp_inner', [(4, 4), (4, 8), (8, 4)])
    # @tune.space(1, 'block_warps_k', [8])
    # @tune.space(1, 'block_warps', [(2, 4)])
    # @tune.space(1, 'warp_outer', [(1, 1)])
    # @tune.space(1, 'warp_mid', [spatial(4, 8)])
    # @tune.space(1, 'warp_inner', [(4, 4)])
    def cuda_schedule_matmul(
            self,
            block_warps_k=8,
            block_warps=(4, 2),
            warp_outer=(2, 2),
            atom_layout=TaskMapping.row_major([4, 8]),
            warp_mid = spatial(4, 8),
            #atom_layout_name='row_4x8',
            warp_inner=(4, 4),
            dtype='float32',
            ) -> IRModule:
        task = self
        local_layout = DataLayout.local
        row_major = DataLayout.row_major
        col_major = DataLayout.column_major
        # Fixed params
        warp_k = 1

        # Native layouts, to be tuned
        # warp_inner = (4, 4)
        # warp_mid = (4, 8)
        # warp_outer = (2, 2)
        # block_warps_k = 8
        # block_warps = (4, 2)



        # Task Layouts
        warp_inner_layout = repeat(*warp_inner, attrs='u+u+')
        #warp_mid_layout = spatial(*warp_mid)
        warp_mid_layout = warp_mid
        warp_outer_layout = repeat(*warp_outer, attrs='u+u+')
        warp_layout = warp_outer_layout * warp_mid_layout * warp_inner_layout
        block_warps_layout = spatial(*block_warps)
        block_layout = block_warps_layout * warp_layout
        block_k = block_warps_k * warp_k  # 8 * 1 = 8
        warp_mid_shape = warp_mid_layout.task_shape  # 4x8
        block_shape = block_layout.task_shape  # 128x128
        warp_size = 32
        block_size = block_layout.num_workers  # 256

        lines = block_size // block_k  # 256 // 8 = 32
        # spatial(32x8) x repeat(4x1)
        a_g2s_layout = spatial(lines, block_k) * \
            repeat(block_shape[0] // lines, 1, attrs='u+.')
        # repeat(1x4) x spatial(8x32)
        b_g2s_layout = repeat(
            1, block_shape[1] // lines, attrs='.u+') * spatial(block_k, lines)
        a_s2r_layout = (     # spatial(4x2) x repeat(2x1) x spatial(4x8) x repeat(4x1)
            block_warps_layout
            * repeat(warp_outer[0], warp_k, attrs='u+u+')
            * warp_mid_layout
            * repeat(warp_inner[0], warp_k, attrs='u+u+')
        )
        b_s2r_layout = (     # spatial(4x2) x repeat(1x2) x spatial(4x8) x repeat(1x4)
            block_warps_layout
            * repeat(warp_k, warp_outer[1], attrs='u+u+')
            * warp_mid_layout
            * repeat(warp_k, warp_inner[1], attrs='u+u+')
        )

        # Data Layouts
        regs_a_layout = (
            local_layout((block_warps[0], 1))      # 4x1
            * col_major((warp_outer[0], warp_k))   # 2x1
            * local_layout((warp_mid_shape[0], 1))  # 4x1
            * row_major((warp_inner[0], 1))        # 4x1
        )
        regs_a_layout = [2] + regs_a_layout
        regs_b_layout = (
            local_layout((1, block_warps[1]))      # 1x2
            * row_major((warp_k, warp_outer[1]))   # 1x2
            * local_layout((1, warp_mid_shape[1]))  # 1x8
            * row_major((1, warp_inner[1]))        # 1x4
        )
        regs_b_layout = [2] + regs_b_layout
        # local(4x2) x spatial(2x2) x local(4x8) x spatial(4x4)
        regs_c_layout = (
            local_layout(block_warps) * row_major(warp_outer) *
            local_layout(warp_mid_shape) * row_major(warp_inner)
        )
        # local(32x8) x spatial(4x1)
        regs_a_ldg_layout = local_layout((block_size // block_k, block_k)) * row_major(
            (block_shape[0] // (block_size // block_k), 1)
        )
        # spatial(1x4) x local(8x32)
        regs_b_ldg_layout = row_major((1, block_shape[1] // (block_size // block_k))) * local_layout(
            (block_k, block_size // block_k)
        )

        used_smem_bytes_per_block = (
            block_shape[0] + block_shape[1]) * block_k * 2 * data_type('float32').nbytes

        bs = task.attrs['batch_size']
        m_size = task.attrs['m_size']
        n_size = task.attrs['n_size']
        k_size = task.attrs['k_size']
        block_m, block_n = block_shape[0], block_shape[1]
        m_tiles = (m_size + block_m - 1) // block_m
        n_tiles = (n_size + block_n - 1) // block_n
        k_tiles = (k_size + block_k - 1) // block_k

        grid_layout = spatial(m_tiles, n_tiles)

        reserved_regs = 48  # number of reserved registers for intermediate results
        used_num_regs_per_thread = (
            regs_a_layout.size // 2 # account for double buffering
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
            min(
                # hidet.cuda.properties().sharedMemPerMultiprocessor // resident_blocks,
                48 * 1024 // resident_blocks,
                hidet.cuda.properties().sharedMemPerBlock,
            )
            // 128
            * 128
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
                        a: f32[bs, m_size, k_size],
                        regs_a_ldg: TensorType(dtype='float32', layout=regs_a_ldg_layout),
                        offset_m: i32,
                        offset_k: i32,
                        first_k_tile_size: i32
            ):
                gmem_a = a[blockIdx.y, offset_m:, offset_k:]
                for i, k in a_g2s_layout.on(threadIdx.x):
                    k_predicate = ((first_k_tile_size == 0) or k < first_k_tile_size)
                    if offset_m + i < m_size and k_predicate:
                        regs_a_ldg[i, k] = gmem_a.read([i,k], protected=False)
                    else:
                        regs_a_ldg[i, k] = 0.0

            @hidet.script
            def copy_a_r2s(
                        regs_a_ldg: TensorType(dtype='float32', layout=regs_a_ldg_layout),
                        smem_a: TensorType(dtype='float32', layout=
                            StridesLayout.from_shape([2, block_m, block_k], perm=[0, 2, 1])),
                        buffer_idx: i32
            ):
                for i, k in a_g2s_layout.on(threadIdx.x):
                    smem_a[buffer_idx, i, k] = regs_a_ldg[i, k]

            @hidet.script
            def copy_a_s2r(
                smem_a: TensorType(dtype='float32', layout=
                    StridesLayout.from_shape([2, block_m, block_k], perm=[0, 2, 1])),
                regs_a: TensorType(dtype='float32', layout=regs_a_layout),
                smem_buffer_idx: i32,
                regs_buffer_idx: i32,
                k_frag_idx: i32
            ):
                smem_a_start = smem_a[smem_buffer_idx, :, k_frag_idx:]
                for i, k in a_s2r_layout.on(threadIdx.x):
                    regs_a[regs_buffer_idx, i, k] = smem_a_start[i, 0]


            @hidet.script
            def copy_b_g2r(
                        b: f32[bs, k_size, n_size],
                        regs_b_ldg: TensorType(dtype='float32', layout=regs_b_ldg_layout),
                        offset_k: i32,
                        offset_n: i32,
                        first_k_tile_size: i32
            ):
                gmem_b = b[blockIdx.y, offset_k:, offset_n:]
                for k, j in b_g2s_layout.on(threadIdx.x):
                    k_predicate = ((first_k_tile_size == 0) or k < first_k_tile_size)
                    if offset_n + j < n_size and k_predicate:
                        regs_b_ldg[k, j] = gmem_b.read([k, j], protected=False)
                    else:
                        regs_b_ldg[k, j] = 0.0
            

            @hidet.script
            def copy_b_r2s(
                        regs_b_ldg: TensorType(dtype='float32', layout=regs_b_ldg_layout),
                        smem_b: TensorType(dtype='float32', layout=
                            StridesLayout.from_shape([2, block_k, block_n], perm=[0, 1, 2])),
                        buffer_idx: i32
            ):
                for k, j in b_g2s_layout.on(threadIdx.x):
                    smem_b[buffer_idx, k, j] = regs_b_ldg[k, j]
        
            @hidet.script
            def copy_b_s2r(
                smem_b: TensorType(dtype='float32', layout=
                    StridesLayout.from_shape([2, block_k, block_n], perm=[0, 1, 2])),
                regs_b: TensorType(dtype='float32', layout=regs_b_layout),
                smem_buffer_idx: i32,
                regs_buffer_idx: i32,
                k_frag_idx: i32
            ):
                smem_b_start = smem_b[smem_buffer_idx, k_frag_idx:, :]
                for k, j in b_s2r_layout.on(threadIdx.x):
                    regs_b[regs_buffer_idx, k, j] = smem_b_start[0, j]
            
            @hidet.script
            def copy_c_r2g(
                regs_c: TensorType(dtype='float32', layout=regs_c_layout),
                c: f32[bs, m_size, n_size],
                offset_m: i32,
                offset_n: i32
            ):
                gmem_c = c[blockIdx.y, offset_m:, offset_n:]
                for i, j in block_layout.on(threadIdx.x):
                    if offset_m + i < m_size and offset_n + j < n_size:
                        gmem_c.write([i, j], regs_c[i, j], protected=False)

            @hidet.script
            def mma(
                regs_a: TensorType(dtype='float32', layout=regs_a_layout),
                regs_b: TensorType(dtype='float32', layout=regs_b_layout),
                regs_c: TensorType(dtype='float32', layout=regs_c_layout),
                buffer_idx: i32
            ):
                for i, j in block_layout.on(threadIdx.x):
                    for k in range(warp_k):
                        regs_c[i, j] += (regs_a[buffer_idx, i, k] * regs_b[buffer_idx, k, j])


            @hidet.script
            def mm_kernel(
                a: f32[bs, m_size, k_size],
                b: f32[bs, k_size, n_size],
                c: f32[bs, m_size, n_size]
            ):
                attr.cuda_grid_dim = (m_tiles * n_tiles, bs)
                attr.cuda_block_dim = block_size
                attr.cuda_dynamic_smem_bytes = cuda_dynamic_smem_bytes
                attr.cuda_min_blocks = min_thread_blocks

                offset_m, offset_n = blockIdx.x // n_tiles * \
                    block_m, blockIdx.x % n_tiles * block_n

                smem = tensor('shared', 'int8', shape=[used_smem_bytes_per_block])

                smem_a = tensor_pointer('float32', layout=StridesLayout.from_shape(
                    [2, block_m, block_k], perm=[0, 2, 1]))
                smem_b = tensor_pointer('float32', layout=StridesLayout.from_shape(
                    [2, block_k, block_n], perm=[0, 1, 2]))
                smem_a_bytes = smem_a.type.tensor_type.storage_bytes()
                smem_a = ~smem[0]
                smem_b = ~smem[smem_a_bytes]

                regs_a = tensor('register', 'float32', layout=regs_a_layout)
                regs_b = tensor('register', 'float32', layout=regs_b_layout)
                regs_c = tensor('register', 'float32', layout=regs_c_layout)
                regs_a_ldg = tensor('register', 'float32',
                                    layout=regs_a_ldg_layout)
                regs_b_ldg = tensor('register', 'float32',
                                layout=regs_b_ldg_layout)

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
                    # pragma unroll
                    for k_frag in grid(block_warps_k, attrs='u+'):
                        if k_frag == block_warps_k - 1:
                            # Store next AB tile from local into shared
                            copy_a_r2s(regs_a_ldg, smem_a, (k + 1) % 2)
                            copy_b_r2s(regs_b_ldg, smem_b, (k + 1) % 2)
                            syncthreads()
                            # Load next k-fragment (from next k-tile) from shared to local
                            copy_a_s2r(smem_a, regs_a, (k + 1) % 2,(k_frag + 1) % 2, 0)
                            copy_b_s2r(smem_b, regs_b, (k + 1) % 2,(k_frag + 1) % 2, 0)
                            pass
                        else:
                            # Load next k-fragment from shared to local
                            copy_a_s2r(smem_a, regs_a, k % 2,(k_frag + 1) % 2, k_frag + 1)
                            copy_b_s2r(smem_b, regs_b, k % 2,(k_frag + 1) % 2, k_frag + 1)
                            pass
                        if k_frag == 0:
                            # Load next AB tile from global into local
                            copy_a_g2r(a, regs_a_ldg, offset_m, offset_k, 0)
                            copy_b_g2r(b, regs_b_ldg, offset_k, offset_n, 0)
                            pass
                        # Perform MMA
                        mma(regs_a, regs_b, regs_c, k_frag % 2)
                # Perform MMA for last k-tile
                last_k = k_tiles - 1
                for k_frag in grid(block_warps_k, attrs='u+'):
                    if k_frag < block_warps_k - 1:
                        # Load next k-fragment from shared to local
                        copy_a_s2r(smem_a, regs_a, (last_k) % 2,(k_frag + 1) % 2, k_frag + 1)
                        copy_b_s2r(smem_b, regs_b, (last_k) % 2,(k_frag + 1) % 2, k_frag + 1)
                        pass
                    # Perform MMA
                    mma(regs_a, regs_b, regs_c, (k_frag) % 2)

                # Store results from regs_c into C
                copy_c_r2g(regs_c, c, offset_m, offset_n) 

        ir_module = module.ir_module()
        #add_packed_func(ir_module, func=mm_kernel, pack_func_name=task.name)
        return ir_module


class MatMulOp(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(inputs=[x, y], task=MatMulTask('matmul', input_like(
            x, 'x'), input_like(y, 'y')), attributes={})


hidet.option.search_space(2)
hidet.option.save_lower_ir(True)
hidet.option.cache_dir('.')

a = hidet.randn([1, 4096, 4096], dtype='float32', device='cuda')
b = hidet.randn([1, 4096, 4096], dtype='float32', device='cuda')

numpy_c = np.matmul(a.cpu().numpy(), b.cpu().numpy())

print("Ref: ", BatchMatmulOp(a, b).latency())
c = BatchMatmulOp(a, b).get_output(0)
np.testing.assert_allclose(actual=c.cpu().numpy(),
                           desired=numpy_c, atol=1e-3, rtol=1e-3)
print("Mine: ", MatMulOp(a, b).latency())
c = MatMulOp(a, b).get_output(0)
np.testing.assert_allclose(actual=c.cpu().numpy(),
                           desired=numpy_c, atol=1e-3, rtol=1e-3)

