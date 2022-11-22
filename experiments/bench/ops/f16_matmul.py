import numpy as np
import numpy.testing
from hidet.ir.func import IRModule
from hidet.ir.compute import TensorNode
from hidet.graph import Operator, Tensor
from hidet.graph.ops.definitions.matmul.batch_matmul import BatchMatmulTask
from hidet.graph.ops.definitions.utils import input_like
import hidet

hidet.option.cache_dir('./outs/cache')
hidet.option.save_lower_ir()


class BatchMatmulF16Task(BatchMatmulTask):
    def __init__(self, a: TensorNode, b: TensorNode):
        super().__init__(a, b)
        self.name = 'batch_matmul_f16'

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return False

    def implement_cuda(self, working_dir: str) -> IRModule:
        batch_size, m_size, n_size, k_size = (self.attributes['batch_size'],
                                              self.attributes['m_size'],
                                              self.attributes['n_size'],
                                              self.attributes['k_size'])
        return self.schedule(batch_size, m_size, n_size, k_size)

    def schedule(self, batch_size, m_size, n_size, k_size) -> IRModule:
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

        # optimize for 128x768x3072
        # mma_config = MmaConfig.m16n8k16_f16_f16()
        # block_m, block_n, block_k = 128, 128, 64
        # warp_m, warp_n, warp_k = 64, 32, 64
        # mma_config = MmaConfig.m16n8k8_f16_f16()
        mma_config = MmaConfig.m16n8k16_f16_f16()
        block_m, block_n, block_k = 64, 128, 16
        warp_m, warp_n, warp_k = 32, 64, 16

        # block_m, block_n, block_k = 128, 128, 16
        # warp_m, warp_n, warp_k = 64, 64, 16
        warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
        mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
        mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
        threads = warp_count_m * warp_count_n * warp_count_k * 32
        assert block_m % warp_m == block_n % warp_n == block_k % warp_k == 0
        assert warp_m % mma_m == warp_n % mma_n == warp_k % mma_k == 0
        assert block_n % 64 == 0
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
                c: float16[batch_size, m_size, n_size]
            ):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
                gmem_c = c[blockIdx.z, offset_m:, offset_n:]
                for k_round in range(warp_count_k):
                    for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                        if wk == k_round:
                            for mi, mj in repeat(mma_count_m, mma_count_n).on(0):
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
                a: float16[batch_size, m_size, k_size],
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
                    cp_async_wait_all()
                    if k0 == 0 and i < 2 and k < 2:
                        printf(r'gmem_a[%d, %d] = %f, smem_a[%d, %d] = %f src_size %d %d\n', i, k, cast(gmem_a[i, k], float32), i, k, cast(smem_a[i, k], float32), src_size, cast(cast(~gmem_a[i, k], 'uint64') % 16, 'int32'))

            @hidet.script
            def load_smem_b(
                k0: int,
                b: float16[batch_size, k_size, n_size],
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
                    if k0 == 0 and k < 2 and j < 2:
                        printf(r'gmem_b[%d, %d] = %f, smem_b[%d, %d] = %f src_size %d\n', k, j, cast(gmem_b[k, j], float32), k, j, cast(smem_b[k, j], float32), src_size)

            @hidet.script
            def gemm_mma_cp_async_ldmatrix_opt_grid(
                a: float16[batch_size, m_size, k_size],
                b: float16[batch_size, k_size, n_size],
                c: float16[batch_size, m_size, n_size]
            ):
                # matrix multiplication, using mma instruction
                attr.cuda_grid_dim = (m_size + block_m - 1) // block_m, (n_size + block_n - 1) // block_n, batch_size
                attr.cuda_block_dim = threads
                # the second 2 means '2 bytes per float16'
                attr.cuda_dynamic_smem_bytes = 2 * (block_m + block_n) * block_k * 2
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
                if threadIdx.x == 0:
                    printf(r"%d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z)
                    for i, k in grid(m_size, k_size):
                        printf(r"a[%d, %d] = %f (%f)\n", i, k, cast(a[blockIdx.z, i, k], float32), cast(smem_a[0, i, k], float32))
                    for k, j in grid(k_size, n_size):
                        printf(r"b[%d, %d] = %f (%f)\n", k, j, cast(b[blockIdx.z, k, j], float32), cast(smem_b[0, k, j], float32))
                syncthreads()
                for k0 in range((k_size + block_k - 1) // block_k):
                    load_smem_a(k0 + 1, a, ~smem_a[(k0 + 1) % 2, 0, 0])
                    load_smem_b(k0 + 1, b, ~smem_b[(k0 + 1) % 2, 0, 0])
                    for mi in range(mma_count_m):
                        load_regs_a(mi, 0, ~smem_a[k0 % 2, 0, 0], ~regs_a[0, mi, 0])
                    for mj in range(mma_count_n):
                        load_regs_b(mj, 0, ~smem_b[k0 % 2, 0, 0], ~regs_b[0, mj, 0])
                    if threadIdx.x == 0:
                        printf(r"shared a:\n")
                        for i in range(mma_m):
                            for k in range(mma_k):
                                printf(r"%.2f ", cast(smem_a[0, i, k], float32))
                            printf(r"\n")
                        printf(r"shared b:\n")
                        for k in range(mma_k):
                            for j in range(mma_n):
                                printf(r"%.2f ", cast(smem_b[0, k, j], float32))
                            printf(r"\n")
                    if threadIdx.x / 32 == 0:
                        print_segment_a(mma_config, ~regs_a[0, 0, 0], worker_id=threadIdx.x % 32)
                        print_segment_b(mma_config, ~regs_b[0, 0, 0], worker_id=threadIdx.x % 32)
                    for mk in range(mma_count_k):
                        if mk + 1 < mma_count_k:
                            for mi in range(mma_count_m):
                                load_regs_a(mi, mk + 1, ~smem_a[k0 % 2, 0, 0], ~regs_a[(mk + 1) % 2, mi, 0])
                            for mj in range(mma_count_n):
                                load_regs_b(mj, mk + 1, ~smem_b[k0 % 2, 0, 0], ~regs_b[(mk + 1) % 2, mj, 0])
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            warp_mma(~regs_a[mk % 2, mi, 0], ~regs_b[mk % 2, mj, 0], ~regs_c[mi, mj, 0])
                            if threadIdx.x < 32 and mk == 0 and mi == 0 and mj == 0:
                                print_segment_c(mma_config, ~regs_c[mi, mj, 0], worker_id=threadIdx.x % 32)
                            break
                    cp_async_wait_all()
                    syncthreads()
                store_c(regs_c, c)

        ir_module = module.ir_module()
        add_packed_func(ir_module, gemm_mma_cp_async_ldmatrix_opt_grid, self.name)

        return ir_module


class BatchMatmulF16Op(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        super().__init__(inputs=[a, b], task=BatchMatmulF16Task(input_like(a, 'a'), input_like(b, 'b')))


def batch_matmul_f16(a: Tensor, b: Tensor) -> Tensor:
    if a.dtype != 'float16' or b.dtype != 'float16':
        raise ValueError('BatchMatmulF16Op only support float16, got {} and {}'.format(a.dtype, b.dtype))
    return BatchMatmulF16Op(a, b).get_output(0)


def main():
    numpy.set_printoptions(linewidth=180)
    n = 2
    a = hidet.ones([1, n, n], dtype='float16')
    b = hidet.ones([1, n, n], dtype='float16')
    c1 = batch_matmul_f16(a, b).numpy()
    print(c1)
    c2 = numpy.matmul(a.numpy(), b.numpy())
    print(c2)
    # print(c2 - c1)
    numpy.testing.assert_allclose(c1, c2, rtol=2e-2, atol=2e-2)


if __name__ == '__main__':
    main()
