import hidet
import numpy as np
from typing import List, Callable, Any, Union, Optional, Dict
from hidet.ir import IRModule
from hidet.ir.compute import ReduceOperation, reduce
from hidet.ir.layout import DataLayout, StridesLayout
from hidet.ir.mapping import TaskMapping
from hidet.ir.type import data_type, TensorType
from hidet.graph.ops.definitions.matmul.batch_matmul import BatchMatmulOp
from hidet.transforms.tools import fuse_and_pack
from hidet.lang import f16, f32, i32, spatial, repeat, tensor, attr, grid, printf, cast
from hidet.lang.mapping import repeat, spatial
from hidet.lang.cuda import blockIdx, threadIdx, syncthreads
from hidet.transforms.tools import add_packed_func
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, InverseMap, compute, input_like, broadcast_shape, broadcast_shapes, broadcast_indices


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

    def implement_cuda(self, workding_dir: str) -> IRModule:
        return cuda_schedule_matmul(self)
        # return cuda_schedule_matmul_smem(self)


def cuda_schedule_matmul(task: MatMulTask) -> IRModule:
    local_layout = DataLayout.local
    row_major = DataLayout.row_major
    col_major = DataLayout.column_major

    # Native layouts, to be tuned
    warp_inner = (4, 4)
    warp_mid = (4, 8)
    warp_outer = (2, 2)
    block_warps_k = 8
    warp_k = 1
    block_warps = (4, 2)

    # Task Layouts
    warp_inner_layout = repeat(*warp_inner)
    warp_mid_layout = spatial(*warp_mid)
    warp_outer_layout = repeat(*warp_outer)
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
        repeat(block_shape[0] // lines, 1)
    # repeat(1x4) x spatial(8x32)
    b_g2s_layout = repeat(
        1, block_shape[1] // lines) * spatial(block_k, lines)
    a_s2r_layout = (     # spatial(4x2) x repeat(2x1) x spatial(4x8) x repeat(4x1)
        block_warps_layout
        * repeat(warp_outer[0], warp_k)
        * warp_mid_layout
        * repeat(warp_inner[0], warp_k)
    )
    b_s2r_layout = (     # spatial(4x2) x repeat(1x2) x spatial(4x8) x repeat(1x4)
        block_warps_layout
        * repeat(warp_k, warp_outer[1])
        * warp_mid_layout
        * repeat(warp_k, warp_inner[1])
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

    bs = task.attributes['batch_size']
    m_size = task.attributes['m_size']
    n_size = task.attributes['n_size']
    k_size = task.attributes['k_size']
    block_m, block_n = block_shape[0], block_shape[1]
    m_tiles = (m_size + block_m - 1) // block_m
    n_tiles = (n_size + block_n - 1) // block_n
    k_tiles = (k_size + block_k - 1) // block_k

    grid_layout = spatial(m_tiles, n_tiles)

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
                    regs_a_ldg[i, k] = gmem_a.read([i,k], protected=True)
                else:
                    regs_a_ldg[i, k] = 0.0

        @hidet.script
        def copy_a_r2s(
                    regs_a_ldg: TensorType(dtype='float32', layout=regs_a_ldg_layout),
                    smem_a: f32[2, block_m, block_k],
                    buffer_idx: i32
        ):
            for i, k in a_g2s_layout.on(threadIdx.x):
                smem_a[buffer_idx, i, k] = regs_a_ldg[i, k]

        @hidet.script
        def copy_a_s2r(
            smem_a: f32[2, block_m, block_k],
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
                    regs_b_ldg[k, j] = gmem_b.read([k, j], protected=True)
                else:
                    regs_b_ldg[k, j] = 0.0
        

        @hidet.script
        def copy_b_r2s(
                    regs_b_ldg: TensorType(dtype='float32', layout=regs_b_ldg_layout),
                    smem_b: f32[2, block_k, block_n],
                    buffer_idx: i32
        ):
            for k, j in b_g2s_layout.on(threadIdx.x):
                smem_b[buffer_idx, k, j] = regs_b_ldg[k, j]
    
        @hidet.script
        def copy_b_s2r(
            smem_b: f32[2, block_k, block_n],
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
                    gmem_c.write([i, j], regs_c[i, j], protected=True)

        @hidet.script
        def mma(
            regs_a: TensorType(dtype='float32', layout=regs_a_layout),
            regs_b: TensorType(dtype='float32', layout=regs_b_layout),
            regs_c: TensorType(dtype='float32', layout=regs_c_layout),
            buffer_idx: i32
        ):
            for i, j in block_layout.on(threadIdx.x):
                for k in range(warp_k):
                    regs_c[i, j] += regs_a[buffer_idx, i, k] * regs_b[buffer_idx, k, j]


        @hidet.script
        def mm_kernel(
            a: f32[bs, m_size, k_size],
            b: f32[bs, k_size, n_size],
            c: f32[bs, m_size, n_size]
        ):
            attr.cuda_grid_dim = (m_tiles * n_tiles, bs)
            attr.cuda_block_dim = block_size

            offset_m, offset_n = blockIdx.x // n_tiles * \
                block_m, blockIdx.x % n_tiles * block_n

            smem_a = tensor('shared', 'float32', layout=StridesLayout.from_shape(
                [2, block_m, block_k], perm=[0, 2, 1]))
            smem_b = tensor('shared', 'float32', layout=StridesLayout.from_shape(
                [2, block_k, block_n], perm=[0, 1, 2]))
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
                for k_frag in range(block_warps_k):
                    if k_frag == block_warps_k - 1:
                        # Store next AB tile from local into shared
                        copy_a_r2s(regs_a_ldg, smem_a, (k + 1) % 2)
                        copy_b_r2s(regs_b_ldg, smem_b, (k + 1) % 2)
                        syncthreads()
                        # Load next k-fragment (from next k-tile) from shared to local
                        copy_a_s2r(smem_a, regs_a, (k + 1) % 2,(k_frag + 1) % 2, 0)
                        copy_b_s2r(smem_b, regs_b, (k + 1) % 2,(k_frag + 1) % 2, 0)
                    else:
                        # Load next k-fragment from shared to local
                        copy_a_s2r(smem_a, regs_a, k % 2,(k_frag + 1) % 2, k_frag + 1)
                        copy_b_s2r(smem_b, regs_b, k % 2,(k_frag + 1) % 2, k_frag + 1)
                    if k_frag == 0:
                        # Load next AB tile from global into local
                        copy_a_g2r(a, regs_a_ldg, offset_m, offset_k, 0)
                        copy_b_g2r(b, regs_b_ldg, offset_k, offset_n, 0)
                    # Perform MMA
                    mma(regs_a, regs_b, regs_c, k_frag % 2)
            # Perform MMA for last k-tile
            last_k = k_tiles - 1
            for k_frag in range(block_warps_k):
                if k_frag < block_warps_k - 1:
                    # Load next k-fragment from shared to local
                    copy_a_s2r(smem_a, regs_a, (last_k) % 2,(k_frag + 1) % 2, k_frag + 1)
                    copy_b_s2r(smem_b, regs_b, (last_k) % 2,(k_frag + 1) % 2, k_frag + 1)
                # Perform MMA
                mma(regs_a, regs_b, regs_c, (k_frag) % 2)

            # Store results from regs_c into C
            copy_c_r2g(regs_c, c, offset_m, offset_n) 

    ir_module = module.ir_module()
    add_packed_func(ir_module, func=mm_kernel, pack_func_name=task.name)
    return ir_module


def cuda_schedule_matmul_smem(task: MatMulTask) -> IRModule:
    bs = task.attributes['batch_size']
    m_size = task.attributes['m_size']
    n_size = task.attributes['n_size']
    k_size = task.attributes['k_size']

    local_layout = DataLayout.local
    row_major = DataLayout.row_major

    block_size = 256
    block_m, block_n, block_k = 128, 128, 8
    m_tiles = (m_size + block_m - 1) // block_m
    n_tiles = (n_size + block_n - 1) // block_n
    k_tiles = (k_size + block_k - 1) // block_k

    grid_layout = spatial(m_tiles, n_tiles)
    block_layout = spatial(16, 16)  # 8x8 elements per thread

    with hidet.script_module() as module:
        @hidet.script
        def mm_kernel(
            a: f32[bs, m_size, k_size],
            b: f32[bs, k_size, n_size],
            c: f32[bs, m_size, n_size]
        ):
            attr.cuda_grid_dim = (m_tiles * n_tiles, bs)
            attr.cuda_block_dim = block_size

            offset_m, offset_n = blockIdx.x // n_tiles * \
                block_m, blockIdx.x % n_tiles * block_n

            smem_a = tensor('shared', 'float32', [block_m, block_k])
            smem_b = tensor('shared', 'float32', [block_k, block_n])
            regs_c = tensor('register', 'float32', shape=(
                8, 8), layout=local_layout((1, 1))*row_major((8, 8)))

            for i, j in grid(8, 8):
                regs_c[i, j] = 0.0

            for k in range(k_tiles):
                offset_k = k * block_k
                gmem_a = a[blockIdx.y, offset_m:, offset_k:]
                gmem_b = b[blockIdx.y, offset_k:, offset_n:]
                for i, k in spatial(8, 1).repeat(4, 1).spatial(4, 8).on(threadIdx.x):
                    smem_a[i, k] = gmem_a.read([i, k], protected=True)
                for k, j in spatial(2, 4).repeat(4, 1).spatial(1, 32).on(threadIdx.x):
                    smem_b[k, j] = gmem_b.read([k, j], protected=True)
                syncthreads()
                for i, j in block_layout.repeat(8, 8).on(threadIdx.x):
                    for k_frag in range(block_k):
                        regs_c[i, j] += smem_a[i, k_frag] * smem_b[k_frag, j]
                syncthreads()

            gmem_c = c[blockIdx.y, offset_m:, offset_n:]
            for i, j in block_layout.repeat(8, 8).on(threadIdx.x):
                gmem_c.write([i, j], regs_c[i, j], protected=True)

    ir_module = module.ir_module()
    add_packed_func(ir_module, func=mm_kernel, pack_func_name=task.name)
    return ir_module


class MatMulOp(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(inputs=[x, y], task=MatMulTask('matmul', input_like(
            x, 'x'), input_like(y, 'y')))


hidet.option.search_space(2)
hidet.option.save_lower_ir(True)
hidet.option.cache_dir('.')

a = hidet.randn([1, 4096, 4096], dtype='float32')
b = hidet.randn([1, 4096, 4096], dtype='float32')
r = hidet.randn([2, 4, 6, 8, 10], dtype='float32')

numpy_c = np.matmul(a.numpy(), b.numpy())

print("Mine: ", MatMulOp(a, b).latency())
c = MatMulOp(a, b).get_output(0)
np.testing.assert_allclose(actual=c.cpu().numpy(),
                           desired=numpy_c, atol=1e-3, rtol=1e-3)

print("Ref: ", BatchMatmulOp(a, b).latency())
c = BatchMatmulOp(a, b).get_output(0)
np.testing.assert_allclose(actual=c.cpu().numpy(),
                           desired=numpy_c, atol=1e-3, rtol=1e-3)
