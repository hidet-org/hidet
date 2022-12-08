import hidet
import numpy as np
from typing import List, Callable, Any, Union, Optional, Dict
from hidet.ir import IRModule
from hidet.ir.compute import ReduceOperation, reduce
from hidet.ir.layout import DataLayout, StridesLayout
from hidet.ir.mapping import TaskMapping
from hidet.ir.type import data_type
from hidet.graph.ops.definitions.matmul.batch_matmul import BatchMatmulOp
from hidet.transforms.tools import fuse_and_pack
from hidet.lang import f16, f32, spatial, repeat, tensor, attr, grid, printf, cast
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
        #return cuda_schedule_matmul_smem(self)

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
    warp_inner_layout = repeat(warp_inner)
    warp_mid_layout = spatial(warp_mid)
    warp_outer_layout = repeat(warp_outer)
    warp_layout = warp_outer_layout * warp_mid_layout * warp_inner_layout
    block_warps_layout = spatial(block_warps)
    block_layout = block_warps_layout * warp_layout
    block_k = block_warps_k * warp_k # 8 * 1 = 8
    warp_mid_shape = warp_mid_layout.task_shape # 4x8
    block_shape = block_layout.task_shape # 128x128
    warp_size = 32
    block_size = block_layout.num_workers # 256

    lines = block_size // block_k # 256 // 8 = 32
    # spatial(32x8) x repeat(4x1)
    a_g2s_layout = spatial((lines, block_k)) * repeat((block_shape[0] // lines, 1))
    # repeat(1x4) x spatial(8x32)
    b_g2s_layout = repeat((1,block_shape[1] // lines)) * spatial((block_k, lines))
    a_s2r_layout = (     # spatial(4x2) x repeat(2x1) x spatial(4x8) x repeat(4x1)
        block_warps_layout
        * repeat([warp_outer[0], warp_k])
        * warp_mid_layout
        * repeat([warp_inner[0], warp_k])
    ).projection({1: 0})
    b_s2r_layout = (     # spatial(4x2) x repeat(1x2) x spatial(4x8) x repeat(1x4)
        block_warps_layout
        * repeat([warp_k, warp_outer[1]])
        * warp_mid_layout
        * repeat([warp_k, warp_inner[1]])
    ).projection({0: 0})

    # Data Layouts
    regs_a_layout = (
        local_layout((block_warps[0], 1))      # 4x1
        * col_major((warp_outer[0], warp_k))   # 2x1
        * local_layout((warp_mid_shape[0], 1)) # 4x1
        * row_major((warp_inner[0], 1))        # 4x1
    )
    regs_b_layout = (
        local_layout((1, block_warps[1]))      # 1x2
        * row_major((warp_k, warp_outer[1]))   # 1x2
        * local_layout((1, warp_mid_shape[1])) # 1x8
        * row_major((1, warp_inner[1]))        # 1x4
    )
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
        def copy(src, dst, layout, src_predicate=None, dst_predicate=None, default_value=0.0):
            for indices in layout(threadIdx.x):
                if dst_predicate and not dst_predicate(*indices):
                    continue
                if src_predicate and not src_predicate(*indices):
                    value = default_value
                else:
                    value = src.read(indices, protected=True)
                dst[indices] = src.read(indices, protected=True)
            return


        @hidet.script
        def mm_kernel(
            a: f32[bs, m_size, k_size],
            b: f32[bs, k_size, n_size],
            c: f32[bs, m_size, n_size]
        ):
            attr.cuda_grid_dim = (m_tiles * n_tiles, bs)
            attr.cuda_block_dim = block_size

            offset_m, offset_n = blockIdx.x // n_tiles * block_m, blockIdx.x % n_tiles * block_n

            smem_a = tensor('shared', 'float32', layout=StridesLayout.from_shape(
                [2, block_m, block_k], perm=[0, 2, 1]))
            smem_b = tensor('shared', 'float32', layout=StridesLayout.from_shape(
                [2, block_k, block_n], perm=[0, 1, 2]))
            regs_a = tensor('register', 'float32', layout=[2] + regs_a_layout)
            regs_b = tensor('register', 'float32', layout=[2] + regs_b_layout)
            regs_c = tensor('register', 'float32', layout=regs_c_layout)
            regs_a_ldg = tensor('register', 'float32', layout=regs_a_ldg_layout)
            regs_b_ldg = tensor('register', 'float32', layout=regs_b_ldg_layout)

            
            # Copy first k-tile from global to shared
            first_k_tile_size = k_size - (k_tiles - 1) * block_k
            gmem_a = a[blockIdx.y, offset_m:, :]
            gmem_b = b[blockIdx.y, :, offset_n:]
            copy(gmem_a, regs_a_ldg, regs_a_ldg_layout, src_predicate=lambda i, k:
                 offset_m + i < m_size and k < first_k_tile_size)
            copy(regs_a_ldg, smem_a[0], a_g2s_layout)
            copy(gmem_b, regs_b_ldg, regs_b_ldg_layout, src_predicate=lambda k, j:
                 offset_n + j < n_size and k < first_k_tile_size)
            copy(regs_b_ldg, smem_b[0], b_g2s_layout)
            syncthreads()

            # Copy first k-frag within first k-tile from shared to local
            copy(smem_a[0], regs_a[0], a_s2r_layout)
            copy(smem_b[0], regs_b[0], b_s2r_layout)
            syncthreads()

            # Initialize regs C
            for i, j in block_layout(threadIdx.x):
                regs_c[i, j] = 0.0

            # Main k-tile loop
            for k in range(k_tiles - 1):
                offset_k = k * block_k + first_k_tile_size
                for k_frag in range(block_warps_k):
                    if k_frag == block_warps_k - 1:
                        # Store next AB tile from local into shared
                        pass
                    if k_frag == 0:
                        # Load next AB tile from global into local
                        pass
                    # Perform MMA

            # Perform MMA for last k-tile
            # Store results from regs_c into C


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

            offset_m, offset_n = blockIdx.x // n_tiles * block_m, blockIdx.x % n_tiles * block_n

            smem_a = tensor('shared', 'float32', [block_m, block_k])
            smem_b = tensor('shared', 'float32', [block_k, block_n])
            regs_c = tensor('register', 'float32', shape=(8,8), layout=local_layout((1,1))*row_major((8,8)))

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
                # for i in row:
                #   for j in col:
                #       regs_c[i,j] += a[batch, arow(i), acol(k)] * b[batch, brow(k), bcol(j)]
                for i, j in block_layout.repeat(8, 8).on(threadIdx.x):
                    for k_frag in range(block_k):
                        regs_c[i, j] += smem_a[i, k_frag] * smem_b[k_frag, j]
                syncthreads()

            gmem_c = c[blockIdx.y, offset_m:, offset_n:]
            for i, j in block_layout.repeat(8,8).on(threadIdx.x):
                gmem_c.write([i, j], regs_c[i, j], protected=True)

    ir_module = module.ir_module()
    add_packed_func(ir_module, func=mm_kernel, pack_func_name=task.name)
    return ir_module


class MatMulOp(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(inputs=[x, y], task=MatMulTask('matmul', input_like(
            x, 'x'), input_like(y, 'y')))


hidet.option.search_space(0)
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

# c = hidet.ops.reduce_sum(r,[1,4])
# print(c)
