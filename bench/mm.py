import hidet
import numpy as np
from typing import List, Callable, Any, Union, Optional, Dict
from hidet.ir import IRModule
from hidet.ir.compute import ReduceOperation, reduce
from hidet.ir.layout import DataLayout
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
        #return cuda_schedule_matmul(self)
        return cuda_schedule_matmul_smem(self)

def cuda_schedule_matmul(task: MatMulTask) -> IRModule:
    local_layout = DataLayout.local
    row_major = DataLayout.row_major

    bs = task.attributes['batch_size']
    m_size = task.attributes['m_size']
    n_size = task.attributes['n_size']
    k_size = task.attributes['k_size']

    warp_inner = (4, 4)
    warp_outer = (2, 2)

    block_size = 256
    block_m, block_n, block_k = 128, 128, 8
    m_tiles = (m_size + block_m - 1) // block_m
    n_tiles = (n_size + block_n - 1) // block_n
    k_tiles = (k_size + block_k - 1) // block_k

    grid_layout = spatial(m_tiles, n_tiles)
    block_layout = spatial(16, 16)  # 8x8 elements per thread

    with hidet.script_module() as module:
        @hidet.script
        def load_regs_a(
                smem_a: f32[block_m, block_k],
                regs_a: f32[8, 1]
        ):
            """Load A registers from shared memory."""
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            pass

        @hidet.script
        def load_regs_b(
                smem_b: f32[block_k, block_n],
                regs_b: f32[1, 8]
        ):
            """Load B registers from shared memory."""
            pass

        @hidet.script
        def mma(
                regs_a: f32[8, 1],
                regs_b: f32[1, 8],
                regs_c: f32[8, 8]
        ):
            """Perform matrix multiplication."""
            pass

        @hidet.script
        def store_c(
                regs_c: f32[8, 8],
                c: f32[bs, m_size, n_size]
        ):
            """Store C registers to global memory."""
            pass

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
            # regs_a = tensor('register', 'float32', [8, 1])
            # regs_b = tensor('register', 'float32', [1, 8])
            regs_c = tensor('register', 'float32', [8, 8])

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
                # load_regs_a()
                # load_regs_b()
                # mma(regs_a, regs_b, regs_c)

                # for i in row:
                #   for j in col:
                #       regs_c[i,j] += a[batch, arow(i), acol(k)] * b[batch, brow(k), bcol(j)]
                for i, j in block_layout.repeat(8, 8).on(threadIdx.x):
                    for k_frag in range(block_k):
                        regs_c[i%8, j%8] += smem_a[i, k_frag] * smem_b[k_frag, j]
                syncthreads()

            gmem_c = c[blockIdx.y, offset_m:, offset_n:]
            for i, j in block_layout.repeat(8,8).on(threadIdx.x):
                gmem_c.write([i, j], regs_c[i%8, j%8], protected=True)
            # store_c(regs_c, c)

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

# c = hidet.ops.reduce_sum(r,[1,4])
# print(c)
