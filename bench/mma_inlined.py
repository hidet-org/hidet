import hidet
import numpy as np
from typing import List, Callable, Any, Union, Optional, Dict
from hidet.ir import IRModule
from hidet.ir.compute import ReduceOperation, reduce
from hidet.ir.layout import DataLayout, StridesLayout, data_layout
from hidet.ir.mapping import TaskMapping, auto_map
from hidet.ir.type import data_type, TensorType, TensorPointerType, DataType
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_sync
from hidet.graph.ops.definitions.matmul.batch_matmul import BatchMatmulOp
from hidet.lang import f16, f32, i32, spatial, repeat, tensor, attr, grid, printf, cast, tensor_pointer
from hidet.lang.cuda import blockIdx, threadIdx, syncthreads
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, InverseMap, compute, input_like, broadcast_shape, broadcast_shapes, broadcast_indices
from hidet.graph.ops.definitions.utils import tune
from hidet.graph.ops.schedules.cuda.common import get_transfer_task_map



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


    @tune.space(2, 'block_m', [16, 32, 64, 128, 256])
    @tune.space(2, 'block_n', [8, 16, 32, 64, 128])
    @tune.space(2, 'block_k', [8, 16, 32])
    @tune.space(2, 'warp_m', [16, 32, 64])
    @tune.space(2, 'warp_n', [8, 16, 32, 64])
    @tune.space(2, 'warp_k', [8, 16, 32])
    @tune.space(2, 'mma_config', [MmaConfig.m16n8k8_bf16_f32(), MmaConfig.m16n8k16_bf16_f32(), MmaConfig.m16n8k4_tf32_f32(), MmaConfig.m16n8k8_tf32_f32()])
    # @tune.space(1, 'block_m', [64, 128, 256])
    # @tune.space(1, 'block_n', [64, 128])
    # @tune.space(1, 'block_k', [8, 16, 32])
    # @tune.space(1, 'warp_m', [32, 64])
    # @tune.space(1, 'warp_n', [32, 64])
    # @tune.space(1, 'warp_k', [8, 16, 32])
    # @tune.space(1, 'mma_config', [MmaConfig.m16n8k8_tf32_f32()])
    @tune.space(1, 'block_m', [64])
    @tune.space(1, 'block_n', [128])
    @tune.space(1, 'block_k', [8])
    @tune.space(1, 'warp_m', [32])
    @tune.space(1, 'warp_n', [32])
    @tune.space(1, 'warp_k', [8])
    @tune.space(1, 'mma_config', [MmaConfig.m16n8k4_tf32_f32()])
    def cuda_schedule_matmul(
            self,
            block_m=64,
            block_n=64,
            block_k=16,
            warp_m=32,
            warp_n=32,
            warp_k=16,
            mma_config=MmaConfig.m16n8k8_tf32_f32(),
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
        local_layout = DataLayout.local
        row_major = DataLayout.row_major
        col_major = DataLayout.column_major

        input_a, input_b, input_c = task.inputs[0], task.inputs[1], task.outputs[0]
        input_a_dtype, input_b_dtype, input_c_dtype = [t.type.dtype for t in [input_a, input_b, input_c]]
        mma_type = resolve_mma_type(input_a_dtype, input_b_dtype, input_c_dtype)
        head, input_dtype, output_dtype = mma_type.split('_')  # pylint: disable=unused-variable
        tune.check(mma_config.input_dtype == input_dtype and mma_config.output_dtype == output_dtype)

        block_shape = (block_m, block_n, block_k)
        warp_shape = (warp_m, warp_n, warp_k)
        mma_m, mma_n, mma_k = (mma_config.m, mma_config.n, mma_config.k)

        tune.check(block_m % warp_m == 0 and block_n % warp_n == 0 and block_k % warp_k == 0)
        tune.check(warp_m % mma_m == 0 and warp_n % mma_n == 0 and warp_k % mma_k == 0)
        warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
        mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
        num_warps =  warp_count_m * warp_count_n * warp_count_k
        block_size = num_threads = num_warps * 32
        tune.check(num_threads <= 1024)
        tune.check(block_m * block_k % num_threads == 0)
        tune.check(block_k * block_n % num_threads == 0)

        lines = block_size // block_k  # 256 // 8 = 32

        a_g2s_layout, regs_a_ldg_layout = get_transfer_task_map(
            task_shape=[block_m, block_k], num_workers=num_threads, ranks=[0, 1]
        )
        b_g2s_layout, regs_b_ldg_layout = get_transfer_task_map(
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
            (
                regs_a_layout.size
                + regs_b_layout.size
                + regs_a_ldg_layout.size
                + regs_b_ldg_layout.size
            )
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

        grid_layout = spatial(m_tiles, n_tiles)
        a_dtype = data_type(mma_config.input_dtype)
        b_dtype = data_type(mma_config.input_dtype)
        c_dtype = data_type(mma_config.output_dtype)
        a_zero, b_zero, c_zero = [dtype.zero for dtype in [a_dtype, b_dtype, c_dtype]]

        with hidet.script_module() as module:
            
            @hidet.script
            def copy_c_r2g(
                regs_c: TensorType(dtype=c_dtype, layout=regs_c_layout),
                c: c_dtype[bs, m_size, n_size],
                smem_c: TensorType(dtype=c_dtype, layout=smem_c_layout),
                offset_m: i32,
                offset_n: i32
            ):
                gmem_c = c[blockIdx.y, offset_m:, offset_n:]
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for warp_k_round in range(warp_count_k):
                    for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                        if wk == warp_k_round:
                            for mma_i, mma_j in grid(mma_count_m, mma_count_n):
                                p = 0
                                for i, j in mma_config.c_store_map.on(lane_id):
                                    delta_m = wi * warp_m + mma_i * mma_m + i
                                    delta_n = wj * warp_n + mma_j * mma_n + j
                                    if delta_m < m_size - offset_m and delta_n < n_size - offset_n:
                                        if warp_count_k == 1:
                                            gmem_c.write([delta_m, delta_n], regs_c[mma_i, mma_j, p])
                                        else:
                                            if warp_k_round == 0:
                                                smem_c[delta_m, delta_n] = regs_c[mma_i, mma_j, p]
                                            elif warp_k_round < warp_count_k - 1:
                                                smem_c[delta_m, delta_n] += regs_c[mma_i, mma_j, p]
                                            else:
                                                gmem_c.write([delta_m, delta_n], smem_c[delta_m, delta_n] + regs_c[mma_i, mma_j, p])
                                    p += 1
                    if warp_k_round + 1 != warp_count_k:
                        syncthreads()


            @hidet.script
            def mm_kernel(
                a: f32[bs, m_size, k_size],
                b: f32[bs, k_size, n_size],
                c: f32[bs, m_size, n_size]
            ):
                attr.cuda_grid_dim = (m_tiles * n_tiles, bs)
                attr.cuda_block_dim = block_size

                gmem_a = a[blockIdx.y, :, :]
                gmem_b = b[blockIdx.y, :, :]
                gmem_c = c[blockIdx.y, :, :]

                smem = tensor('shared', 'int8', shape=[smem_storage_nbytes])
                smem_a = tensor_pointer(a_dtype, layout=smem_a_layout)
                smem_b = tensor_pointer(b_dtype, layout=smem_b_layout)
                smem_c = tensor_pointer(c_dtype, layout=smem_c_layout)
                smem_a_bytes = smem_a.type.tensor_type.storage_bytes()
                smem_a = ~smem[0]
                smem_b = ~smem[smem_a_bytes]
                smem_c = ~smem[0]

                regs_a = tensor('register', a_dtype, layout=regs_a_layout)
                regs_b = tensor('register', b_dtype, layout=regs_b_layout)
                regs_c = tensor('register', c_dtype, layout=regs_c_layout)
                regs_a_ldg = tensor('register', a_dtype,
                                    layout=regs_a_ldg_layout)
                regs_b_ldg = tensor('register', b_dtype,
                                layout=regs_b_ldg_layout)
                # Initialize regs C
                for i, j, p in grid(mma_count_m, mma_count_n, mma_config.c_elements):
                    regs_c[i, j, p] = c_zero

                offset_m, offset_n = blockIdx.x // n_tiles * \
                    block_m, blockIdx.x % n_tiles * block_n

                # Copy first k-tile from global to shared
                # copy_a_g2r(a, regs_a_ldg, offset_m, 0)
                gmem_a = a[blockIdx.y, offset_m:, :]
                for i, k in a_g2s_layout.on(threadIdx.x):
                    if offset_m + i < m_size and k < k_size:
                        regs_a_ldg[i, k] = gmem_a.read([i,k], protected=True)
                    else:
                        regs_a_ldg[i, k] = a_zero
                # copy_a_r2s(regs_a_ldg, smem_a, 0)
                for i, k in a_g2s_layout.on(threadIdx.x):
                    smem_a[0, i, k] = regs_a_ldg[i, k]
                # copy_b_g2r(b, regs_b_ldg, 0, offset_n)
                gmem_b = b[blockIdx.y, :, offset_n:]
                for k, j in b_g2s_layout.on(threadIdx.x):
                    if offset_n + j < n_size and k < k_size:
                        regs_b_ldg[k, j] = gmem_b.read([k, j], protected=True)
                    else:
                        regs_b_ldg[k, j] = b_zero
                # copy_b_r2s(regs_b_ldg, smem_b, 0)
                for k, j in b_g2s_layout.on(threadIdx.x):
                    smem_b[0, k, j] = regs_b_ldg[k, j]
                syncthreads()
                # Copy first k-tile from shared to local
                # copy_a_s2r(~smem_a[0, 0, 0], regs_a, 0)
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    for mma_i in grid(mma_count_m, unroll=True):
                        p = 0
                        for i, k in mma_config.a_load_map.on(lane_id):
                            regs_a[0, mma_i, p] = smem_a[0, wi * warp_m + mma_i * mma_m + i, wk * warp_k + k]
                            p += 1
                # copy_b_s2r(~smem_b[0, 0, 0], regs_b, 0)
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    for mma_j in grid(mma_count_n, unroll=True):
                        p = 0
                        for k, j in mma_config.b_load_map.on(lane_id):
                            regs_b[0, mma_j, p] = smem_b[0, wk * warp_k + k, wj * warp_n + mma_j * mma_n + j]
                            p += 1

                for k0 in range(k_tiles):
                    ko = 0
                    if mma_count_k % 2 != 0 and k0 % 2 != 0:
                        ko = 1
                    for k1 in range(mma_count_k):
                        if k1 == 0:
                            offset_k = (k0 + 1) * block_k
                            # copy_a_g2r(a, regs_a_ldg, offset_m, offset_k)
                            gmem_a = a[blockIdx.y, offset_m:, offset_k:]
                            for i, k in a_g2s_layout.on(threadIdx.x):
                                if offset_m + i < m_size and offset_k + k < k_size:
                                    regs_a_ldg[i, k] = gmem_a.read([i,k], protected=True)
                                else:
                                    regs_a_ldg[i, k] = a_zero
                            # copy_b_g2r(b, regs_b_ldg, offset_k, offset_n)
                            gmem_b = b[blockIdx.y, offset_k:, offset_n:]
                            for k, j in b_g2s_layout.on(threadIdx.x):
                                if offset_n + j < n_size and offset_k + k < k_size:
                                    regs_b_ldg[k, j] = gmem_b.read([k, j], protected=True)
                                else:
                                    regs_b_ldg[k, j] = b_zero
                        if k1 == mma_count_k - 1:
                            # copy_a_r2s(regs_a_ldg, smem_a, (k0 + 1) % 2)
                            for i, k in a_g2s_layout.on(threadIdx.x):
                                smem_a[(k0 + 1) % 2, i, k] = regs_a_ldg[i, k]
                            # copy_b_r2s(regs_b_ldg, smem_b, (k0 + 1) % 2)
                            for k, j in b_g2s_layout.on(threadIdx.x):
                                smem_b[(k0 + 1) % 2, k, j] = regs_b_ldg[k, j]
                            syncthreads()
                            # copy_a_s2r(~smem_a[(k0 + 1) % 2, 0, 0], regs_a, (k1 + ko + 1) % 2)
                            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                                for mma_i in grid(mma_count_m, unroll=True):
                                    p = 0
                                    for i, k in mma_config.a_load_map.on(lane_id):
                                        regs_a[(k1 + ko + 1) % 2, mma_i, p] = smem_a[(k0 + 1) % 2, wi * warp_m + mma_i * mma_m + i, wk * warp_k + k]
                                        p += 1
                            # copy_b_s2r(~smem_b[(k0 + 1) % 2, 0, 0], regs_b, (k1 + ko + 1) % 2)
                            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                                for mma_j in grid(mma_count_n, unroll=True):
                                    p = 0
                                    for k, j in mma_config.b_load_map.on(lane_id):
                                        regs_b[(k1 + ko + 1) % 2, mma_j, p] = smem_b[(k0 + 1) % 2, wk * warp_k + k, wj * warp_n + mma_j * mma_n + j]
                                        p += 1
                        else:
                            # copy_a_s2r(~smem_a[k0 % 2, 0, (k1 + 1) * mma_k], regs_a, (k1 + ko + 1) % 2)
                            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                                for mma_i in grid(mma_count_m, unroll=True):
                                    p = 0
                                    for i, k in mma_config.a_load_map.on(lane_id):
                                        regs_a[(k1 + ko + 1) % 2, mma_i, p] = smem_a[k0 % 2, wi * warp_m + mma_i * mma_m + i, (k1 + 1) * mma_k + wk * warp_k + k]
                                        p += 1
                            # copy_b_s2r(~smem_b[k0 % 2, (k1 + 1) * mma_k, 0], regs_b, (k1 + ko + 1) % 2)
                            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                                for mma_j in grid(mma_count_n, unroll=True):
                                    p = 0
                                    for k, j in mma_config.b_load_map.on(lane_id):
                                        regs_b[(k1 + ko + 1) % 2, mma_j, p] = smem_b[k0 % 2, (k1 + 1) * mma_k + wk * warp_k + k, wj * warp_n + mma_j * mma_n + j]
                                        p += 1
                        # mma(regs_a, regs_b, regs_c, (k1 + ko) % 2)
                        for mma_i, mma_j in grid(mma_count_m, mma_count_n, unroll=True):
                            mma_sync(mma_config, ~regs_a[(k1 + ko) % 2, mma_i, 0], ~regs_b[(k1 + ko) % 2, mma_j, 0], ~regs_c[mma_i, mma_j, 0])
                copy_c_r2g(regs_c, c, smem_c, offset_m, offset_n)

        ir_module = module.ir_module()
        return ir_module


class MatMulOp(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(inputs=[x, y], task=MatMulTask('matmul', input_like(
            x, 'x'), input_like(y, 'y')), attributes={})


hidet.option.search_space(1)
hidet.option.save_lower_ir(True)
hidet.option.cache_dir('.')

a = hidet.randn([1, 4096, 4096], dtype='float32', device='cuda')
b = hidet.randn([1, 4096, 4096], dtype='float32', device='cuda')

numpy_c = np.matmul(a.cpu().numpy(), b.cpu().numpy())

# print("Ref: ", BatchMatmulOp(a, b, mma='mma').latency())
# c = BatchMatmulOp(a, b, mma='mma').get_output(0)
# np.testing.assert_allclose(actual=c.cpu().numpy(),
                        #    desired=numpy_c, atol=1e-1, rtol=1e-1)
print("Mine: ", MatMulOp(a, b).latency())
c = MatMulOp(a, b).get_output(0)
np.testing.assert_allclose(actual=c.cpu().numpy(),
                           desired=numpy_c, atol=1e-1, rtol=1e-1)

