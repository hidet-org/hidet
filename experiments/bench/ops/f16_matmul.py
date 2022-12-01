from typing import Optional, List, Tuple

import numpy as np
import numpy.testing
from functools import partial
from hidet.ir.dtypes import float16
from hidet.ir.expr import if_then_else
from hidet.ir.func import IRModule
from hidet.ir.compute import TensorNode
from hidet.graph import Operator, Tensor
from hidet.graph.ops.definitions.matmul.batch_matmul import BatchMatmulTask, BatchMatmulOp
from hidet.graph.ops.definitions.matmul.matmul import MatmulTask
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.definitions.utils import input_like, broadcast_shape, can_broadcast, broadcast_indices
from hidet.graph.ops.schedules import Schedule
from hidet.graph.transforms.resolve_variant import register_resolve_rule, ResolveRule
from hidet.graph.ops.schedules import tune
from hidet.utils.py import is_power_of_two, cdiv, prod
import hidet

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()


class BatchMatmulF16Task(Task):
    def __init__(self, a: TensorNode, b: TensorNode, parallel_k_parts: int = 1):
        a_shape = a.const_shape()
        b_shape = b.const_shape()

        if not a.ttype.dtype == float16 or not b.ttype.dtype == float16:
            raise ValueError('Both inputs must be float16 tensors')

        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError('Matrix multiplication expect at least 2D tensor, got {} and {}'.format(a_shape, b_shape))

        if a_shape[-1] != b_shape[-2]:
            raise ValueError(
                'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                ', got {} and {}'.format(a_shape, b_shape)
            )

        if not can_broadcast(a_shape[:-2], b_shape[:-2]):
            raise ValueError(
                'Matrix multiplication expect tensor A and B with compatible broadcast shape, '
                'got {} and {}'.format(a_shape, b_shape)
            )

        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]
        c_shape = [parallel_k_parts] + broadcast_shape(a_shape[:-2], b_shape[:-2]) + [a_shape[-2], b_shape[-1]]
        k_part_extent = cdiv(k_size, parallel_k_parts)

        c = compute(
            name='c',
            shape=c_shape,
            fcompute=lambda k_part, *indices: reduce(
                shape=[k_part_extent],
                fcompute=lambda k: if_then_else(
                    k_part * k_part_extent + k < k_size,
                    a[broadcast_indices(indices[:-2], a_shape[:-2], c_shape[1:-2]) + [indices[-2], k]] *
                    b[broadcast_indices(indices[:-2], b_shape[:-2], c_shape[1:-2]) + [k, indices[-1]]],
                    float16(0.0)
                ),
                reduce_type='sum'
            )
        )

        super().__init__(
            name='matmul_f16_pk',
            inputs=[a, b],
            outputs=[c],
            attributes={
                'parallel_k_parts': parallel_k_parts
            }
        )

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return False

    # def implement_cuda(self, working_dir: str) -> IRModule:
    #     return tune.tune(self.schedule, task=self, target_device='cuda', working_dir=working_dir)

    # @tune.space(2, 'block_m, block_n, block_k', [[64, 128, 16],
    #                                              [64, 128, 8]])
    # @tune.space(2, 'warp_m, warp_n, warp_k', [[32, 64, 16],
    #                                           [64, 64, 16]])
    @tune.space(2, 'block_m', [32, 64, 128, 256])
    @tune.space(2, 'block_n', [32, 64, 128, 256])
    @tune.space(2, 'block_k', [8, 16, 32, 64, 128])
    @tune.space(2, 'warp_m', [16, 32, 48, 64])
    @tune.space(2, 'warp_n', [16, 32, 48, 64])
    @tune.space(2, 'warp_k', [8, 16, 32, 64])
    # @tune.space(2, 'mma', ['m16n8k8', 'm16n8k16'])
    @tune.space(2, 'mma', ['m16n8k16'])
    # @tune.space(2, 'block_m', [32, 64, 128, 256])
    # @tune.space(2, 'block_n', [32, 64, 128, 256])
    # @tune.space(2, 'block_k', [8, 16, 32, 128])
    # @tune.space(2, 'warp_m', [16, 32, 48])
    # @tune.space(2, 'warp_n', [16, 32, 48])
    # @tune.space(2, 'warp_k', [8, 16, 32])
    # @tune.space(2, 'mma', ['m16n8k16'])
    @tune.space(1, 'block_m', [256])
    @tune.space(1, 'block_n', [64])
    @tune.space(1, 'block_k', [128])
    @tune.space(1, 'warp_m', [64])
    @tune.space(1, 'warp_n', [64])
    @tune.space(1, 'warp_k', [16])
    @tune.space(1, 'mma', ['m16n8k16'])
    def schedule(
        self, block_m=64, block_n=128, block_k=16, warp_m=32, warp_n=64, warp_k=16, mma: str = 'm16n8k16'
    ) -> IRModule:
        from hidet.ir.type import tensor_type
        from hidet.ir.primitives.cuda.mma import print_segment_a, print_segment_b, print_segment_c
        from hidet.lang import spatial, repeat, tensor, attr, col_spatial, view, u32, tensor_pointer, grid, printf, cast
        from hidet.lang.layout import row_layout, col_layout, local_layout, DataLayout
        from hidet.lang.mapping import repeat, spatial
        from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory
        from hidet.lang.cuda import MmaConfig, mma_sync, cp_async, cp_async_wait_all, ldmatrix
        from hidet.lang.cuda import shared_tensor, register_tensor
        from hidet.lang import float16, float32
        from hidet.transforms.tools import add_packed_func

        # input shapes
        node_a, node_b, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape: List[int] = node_a.const_shape()
        b_shape: List[int] = node_b.const_shape()
        c_shape: List[int] = node_c.const_shape()
        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]
        a_head, b_head, c_head = a_shape[:-2], b_shape[:-2], c_shape[:-2]
        k_parts = self.attributes['parallel_k_parts']
        k_part_extent = cdiv(k_size, k_parts)

        # schedule parameters
        mma_configs = {
            'm16n8k8': MmaConfig.m16n8k8_f16_f16(),
            'm16n8k16': MmaConfig.m16n8k16_f16_f16(),
        }
        tune.check(mma in mma_configs)
        mma_config = mma_configs[mma]

        mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
        warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
        mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
        threads = warp_count_m * warp_count_n * warp_count_k * 32
        grid_dim: Tuple[int, int, int] = cdiv(m_size, block_m), cdiv(n_size, block_n), prod(c_head)
        dynamic_smem_bytes = 2 * (block_m + block_n) * block_k * 2

        tune.check(block_m % warp_m == block_n % warp_n == block_k % warp_k == 0, 'warp dims divide block dims')
        tune.check(warp_m % mma_m == warp_n % mma_n == warp_k % mma_k == 0, 'mma dims divide warp dims')
        tune.check(threads <= 1024, 'threads in a block <= 1024')
        tune.check(dynamic_smem_bytes <= 49152, 'dynamic shared memory <= 49152')

        tune.check(block_n % 64 == 0, 'block_n must be multiple of 64, required by async gmem -> smem loading')
        tune.check(block_k % 8 == 0)
        tune.check(is_power_of_two(block_k // 8))
        tune.check(threads % (block_k // 8) == 0)
        tune.check(threads % (block_n // 8) == 0)
        tune.check(block_m % (threads // (block_k // 8)) == 0)
        tune.check(block_k % (threads // (block_n // 8)) == 0)
        smem_a_type = tensor_type(
            'float16', shape=[block_m, block_k],
            layout=row_layout(block_m, block_k // 8).swizzle(1) * row_layout(1, 8)
        )
        smem_b_type = tensor_type(
            'float16', shape=[block_k, block_n],
            layout=row_layout(block_k // 8, block_n // 64) * row_layout(8, 8).swizzle(1) * row_layout(1, 8)
        )
        load_smem_a_map = repeat(block_m // (threads // (block_k // 8)), 1).spatial(
            threads // (block_k // 8), block_k // 8
        )
        load_smem_b_map = repeat(block_k // (threads // (block_n // 8)), 1).spatial(
            threads // (block_n // 8), block_n // 8
        )

        with hidet.script_module() as module:
            @hidet.script
            def load_regs_a(
                mi: int,
                k1: int,
                smem_a: smem_a_type,
                regs_a: float16[mma_config.a_elements]
            ):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    row_addr = ~smem_a[wi * warp_m + mi * mma_m + p, wk * warp_k + k1 * mma_k + q * 8]
                    b32_regs = view(regs_a, u32[4])
                    ldmatrix(
                        regs=[b32_regs[0], b32_regs[1], b32_regs[2], b32_regs[3]],
                        smem_addr=row_addr,
                        shared_space_addr=False,
                        trans=False
                    )

            @hidet.script
            def load_regs_b(
                mj: int,
                k1: int,
                smem_b: smem_b_type,
                regs_b: float16[mma_config.b_elements]
            ):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    # have not used q as we only use the address of the first 16 threads to load 2 of 8x8 f16 matrix.
                    row_addr = ~smem_b[wk * warp_k + k1 * mma_k + p, wj * warp_n + mj * mma_n]
                    regs = view(regs_b, u32[2])
                    ldmatrix(
                        regs=[regs[0], regs[1]],
                        smem_addr=row_addr,
                        trans=True
                    )

            @hidet.script
            def warp_mma(
                regs_a: float16[mma_config.a_elements],
                regs_b: float16[mma_config.b_elements],
                regs_c: float16[mma_config.c_elements]
            ):
                mma_sync(mma_config, regs_a, regs_b, regs_c)

            @hidet.script
            def store_c(
                regs_c: float16[mma_count_m, mma_count_n, mma_config.c_elements],
                c: float16[c_head + [m_size, n_size]]
            ):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
                gmem_c = c[blockIdx.z, offset_m:, offset_n:]
                for k_round in range(warp_count_k):
                    for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                        if wk == k_round:
                            for mi, mj in grid(mma_count_m, mma_count_n):
                                p = 0
                                for i, j in mma_config.c_store_map.on(lane_id):
                                    gmem_c.write(
                                        [wi * warp_m + mi * mma_m + i,
                                         wj * warp_n + mj * mma_n + j],
                                        regs_c[mi, mj, p],
                                        protected=True
                                    )
                                    p += 1

            @hidet.script
            def load_smem_a(
                k0: int,
                a: float16[a_head + [m_size, k_size]],
                smem_a: smem_a_type
            ):
                offset_m, offset_n, offset_k = blockIdx.x * block_m, blockIdx.y * block_n, k0 * block_k
                gmem_a = a[blockIdx.z, offset_m:, offset_k:]
                for i, k_seg in load_smem_a_map.on(threadIdx.x):
                    k = k_seg * 8
                    src_size = 0 if (offset_m + i >= m_size or offset_k + k >= k_size) else min(
                        k_size - (offset_k + k), 8
                    )
                    cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=16, src_size=src_size * 2, cache_level='global')

            @hidet.script
            def load_smem_b(
                k0: int,
                b: float16[b_head + [k_size, n_size]],
                smem_b: smem_b_type
            ):
                offset_m, offset_n, offset_k = blockIdx.x * block_m, blockIdx.y * block_n, k0 * block_k
                gmem_b = b[blockIdx.z, offset_k:, offset_n:]
                for k, j_seg in load_smem_b_map.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = 0 if (offset_k + k >= k_size or offset_n + j >= n_size) else min(
                        n_size - (offset_n + j), 8
                    )
                    cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=16, src_size=src_size * 2, cache_level='global')

            @hidet.script
            def gemm_mma_cp_async_ldmatrix_opt_grid(
                a: float16[a_head + [m_size, k_size]],
                b: float16[b_head + [k_size, n_size]],
                c: float16[c_head + [m_size, n_size]]
            ):
                # matrix multiplication, using mma instruction
                attr.cuda_grid_dim = grid_dim
                attr.cuda_block_dim = threads
                # the second 2 means '2 bytes per float16'
                attr.cuda_dynamic_smem_bytes = dynamic_smem_bytes
                # smem_storage = dyn_smem_storage
                smem_a = tensor_pointer(
                    'float16', shape=[2, block_m, block_k], layout=row_layout(2) + smem_a_type.layout
                )
                smem_b = tensor_pointer(
                    'float16', shape=[2, block_k, block_n], layout=row_layout(2) + smem_b_type.layout
                )
                smem_a = dynamic_shared_memory(byte_offset=0, dtype=float16)
                smem_b = dynamic_shared_memory(byte_offset=2 * block_m * block_k * 2, dtype=float16)
                regs_a = register_tensor(float16, [2, mma_count_m, mma_config.a_elements])
                regs_b = register_tensor(float16, [2, mma_count_n, mma_config.b_elements])
                regs_c = register_tensor(float16, [mma_count_m, mma_count_n, mma_config.c_elements])

                for i, j, p in grid(mma_count_m, mma_count_n, mma_config.c_elements):
                    regs_c[i, j, p] = 0.0

                load_smem_a(0, a, ~smem_a[0, 0, 0])
                load_smem_b(0, b, ~smem_b[0, 0, 0])

                cp_async_wait_all()
                syncthreads()
                for k0 in range((k_size + block_k - 1) // block_k):
                    load_smem_a(k0 + 1, a, ~smem_a[(k0 + 1) % 2, 0, 0])
                    load_smem_b(k0 + 1, b, ~smem_b[(k0 + 1) % 2, 0, 0])
                    for mi in range(mma_count_m):
                        load_regs_a(mi, 0, ~smem_a[k0 % 2, 0, 0], ~regs_a[0, mi, 0])
                    for mj in range(mma_count_n):
                        load_regs_b(mj, 0, ~smem_b[k0 % 2, 0, 0], ~regs_b[0, mj, 0])
                    for mk in range(mma_count_k):
                        if mk + 1 < mma_count_k:
                            for mi in range(mma_count_m):
                                load_regs_a(mi, mk + 1, ~smem_a[k0 % 2, 0, 0], ~regs_a[(mk + 1) % 2, mi, 0])
                            for mj in range(mma_count_n):
                                load_regs_b(mj, mk + 1, ~smem_b[k0 % 2, 0, 0], ~regs_b[(mk + 1) % 2, mj, 0])
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            warp_mma(~regs_a[mk % 2, mi, 0], ~regs_b[mk % 2, mj, 0], ~regs_c[mi, mj, 0])
                    cp_async_wait_all()
                    syncthreads()
                store_c(regs_c, c)

        ir_module = module.ir_module()
        add_packed_func(ir_module, gemm_mma_cp_async_ldmatrix_opt_grid, self.name)

        return ir_module


class BatchMatmulF16Op(Operator):
    def __init__(self, a: Tensor, b: Tensor, parallel_k_parts=1):
        super().__init__(
            inputs=[a, b],
            task=BatchMatmulF16Task(input_like(a, 'a'), input_like(b, 'b'), parallel_k_parts)
        )


def batch_matmul_f16(a: Tensor, b: Tensor, parallel_k_parts=1) -> Tensor:
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise ValueError('Expect 3D tensors')
    if a.shape[-1] % 8 != 0 or b.shape[-1] % 8 != 0:
        raise ValueError('Expect the last dimension of the input tensors to be a multiple of 8')
    if a.dtype != 'float16' or b.dtype != 'float16':
        raise ValueError('BatchMatmulF16Op only support float16, got {} and {}'.format(a.dtype, b.dtype))
    return BatchMatmulF16Op(a, b, parallel_k_parts).get_output(0)


@register_resolve_rule(BatchMatmulOp)
class ResolveBatchMatmulToBatchMatmulF16(ResolveRule):
    def resolve(self, op: BatchMatmulOp) -> Optional[List[Tensor]]:
        a, b = op.inputs
        if a.dtype == 'float16' and b.dtype == 'float16':
            if a.shape[-1] % 8 == 0 and b.shape[-1] % 8 == 0:
                return [batch_matmul_f16(a, b)]
        return None


def main():
    numpy.set_printoptions(linewidth=180)
    hidet.option.search_space(space=0)
    hidet.option.search_space(space=2)
    # hidet.option.search_space(space=1)
    # m, n, k = 128, 768, 32
    # m, n, k = 64, 128, 32
    m, n, k = 1024, 1024, 1024
    # m, n, k = 8, 8, 8
    a = hidet.ones([1, m, k], dtype='float16')
    b = hidet.ones([1, k, n], dtype='float16')
    # c1 = batch_matmul_f16(a, b).numpy()
    c1 = batch_matmul_f16(a, b, parallel_k_parts=32)
    c1 = hidet.ops.reduce_sum(c1, dims=0, keep_dim=False).numpy()
    # print(c1)
    c2 = numpy.matmul(a.numpy(), b.numpy())
    # print(c2)
    # print(c2 - c1)
    numpy.testing.assert_allclose(c1, c2, rtol=2e-2, atol=2e-2)


if __name__ == '__main__':
    main()
