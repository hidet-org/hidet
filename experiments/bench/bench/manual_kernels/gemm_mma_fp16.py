from hidet.runtime import CompiledFunction
import numpy as np
import hidet
from bench.common import BenchResult, benchmark_run


def gemm_mma_fp16_kernel(bs, m_size, n_size, k_size) -> CompiledFunction:
    return gemm_mma_fp16_kernel_v1(bs, m_size, n_size, k_size)


def gemm_mma_fp16_kernel_v1(bs, m_size, n_size, k_size) -> CompiledFunction:
    from hidet.lang import f16, spatial, repeat, tensor, attr
    from hidet.lang.layout import row_layout, col_layout, local_layout
    from hidet.lang.mapping import repeat, spatial
    from hidet.lang.cuda import blockIdx, threadIdx, syncthreads
    from hidet.lang.cuda import MmaConfig, mma_sync

    # optimize for 128x768x3072
    mma_config = MmaConfig.m16n8k16_f16_f16()
    block_m, block_n, block_k = 128, 128, 16
    warp_m, warp_n, warp_k = 64, 64, 16
    warp_count_m, warp_count_n, warp_count_k = 2, 2, 1
    mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
    mma_count_m, mma_count_n, mma_count = 4, 8, 1
    threads = warp_count_m * warp_count_n * warp_count_k * 32

    with hidet.script_module() as module:
        @hidet.script
        def load_regs_a(
                smem_a: tensor('shared', 'float16', [block_m, block_k]),
                regs_a: tensor('register', 'float16', [4, mma_config.a_elements])
        ):
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                for mi in range(mma_count_m):
                    p = 0
                    for i, k in mma_config.a_load_map.on(lane_id):
                        regs_a[mi, p] = smem_a[wi * warp_m + mi * mma_m + i, wk * warp_k + k]
                        p += 1

        @hidet.script
        def load_regs_b(
                smem_b: tensor('shared', 'float16', [block_k, block_n]),
                regs_b: tensor('register', 'float16', [8, mma_config.b_elements])
        ):
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                for mj in range(mma_count_n):
                    p = 0
                    for k, j in mma_config.b_load_map.on(lane_id):
                        regs_b[mj, p] = smem_b[wk * warp_k + k, wj * warp_n + mj * mma_n + j]
                        p += 1

        @hidet.script
        def warp_mma(
                regs_a: tensor('register', 'float16', [4, mma_config.a_elements]),
                regs_b: tensor('register', 'float16', [8, mma_config.b_elements]),
                regs_c: tensor('register', 'float16', [4, 8, mma_config.c_elements])
        ):
            for mi, mj in repeat(mma_count_m, mma_count_n).on(0):
                mma_sync(mma_config, ~regs_a[mi, 0], ~regs_b[mj, 0], ~regs_c[mi, mj, 0])

        @hidet.script
        def store_c(
                regs_c: tensor('register', 'float16', [4, 8, mma_config.c_elements]),
                c: tensor('global', 'float16', [bs, m_size, n_size])
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
                                gmem_c.write([wi * warp_m + mi * mma_m + i,
                                              wj * warp_n + mj * mma_n + j],
                                             regs_c[mi, mj, p],
                                             protected=True)
                                p += 1

        @hidet.script
        def gemm_mma_grid(a: f16[bs, m_size, k_size], b: f16[bs, k_size, n_size], c: f16[bs, m_size, n_size]):
            # matrix multiplication, using mma instruction
            attr.cuda_grid_dim = (m_size + block_m - 1) // block_m, (n_size + block_n - 1) // block_n, bs
            attr.cuda_block_dim = threads
            offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
            smem_a = tensor('shared', 'float16', [block_m, block_k])
            smem_b = tensor('shared', 'float16', [block_k, block_n])
            regs_a = tensor('register', 'float16', [4, mma_config.a_elements])
            regs_b = tensor('register', 'float16', [8, mma_config.b_elements])
            regs_c = tensor('register', 'float16', [4, 8, mma_config.c_elements])

            for i, j, p in repeat(4, 8, mma_config.c_elements).on(0):
                regs_c[i, j, p] = 0.0

            for k0 in range((k_size + block_k - 1) // block_k):
                offset_k = k0 * block_k
                gmem_a = a[blockIdx.z, offset_m:, offset_k:]
                gmem_b = b[blockIdx.z, offset_k:, offset_n:]
                for i, k in repeat(16, 1).spatial(8, 16).on(threadIdx.x):
                    smem_a[i, k] = gmem_a.read([i, k], protected=True)
                for k, j in repeat(16, 1).spatial(1, 128).on(threadIdx.x):
                    smem_b[k, j] = gmem_b.read([k, j], protected=True)
                syncthreads()
                load_regs_a(smem_a, regs_a)
                load_regs_b(smem_b, regs_b)
                warp_mma(regs_a, regs_b, regs_c)
                syncthreads()
            store_c(regs_c, c)

    ir_module = module.ir_module()
    func = hidet.driver.build_ir_module(ir_module, func_name='gemm_mma')
    return func


if __name__ == '__main__':
    bs, m, n, k = 11, 111, 222, 333
    func = gemm_mma_fp16_kernel_v1(bs, m, n, k)
    a = hidet.randint(0, 2, [bs, m, k], 'float16')
    b = hidet.randint(0, 2, [bs, k, n], 'float16')
    c = hidet.zeros([bs, m, n], 'float16')
    func(a, b, c)
    c2 = hidet.ops.matmul(a, b)
    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    # print(a)
    # print(b)
    # print(c)
    # print(c2)
    # print(c-c2)
    np.testing.assert_allclose(actual=c.numpy(), desired=c2.numpy())
