from hidet.runtime import CompiledFunction
import numpy as np
import hidet
from bench.common import BenchResult, benchmark_run


def gemm_mma_fp16_cp_async_multi_stage_kernel(bs, m_size, n_size, k_size) -> CompiledFunction:
    from hidet.lang import f16, f32, spatial, repeat, tensor, attr, printf, cast
    from hidet.lang.layout import row_layout, col_layout, local_layout
    from hidet.lang.mapping import repeat, spatial
    from hidet.lang.cuda import blockIdx, threadIdx, syncthreads
    from hidet.lang.cuda import MmaConfig, mma_sync, cp_async, cp_async_wait_all, cp_async_commit_group, cp_async_wait_group, cvta_generic_to_shared

    # optimize for 128x768x3072
    mma_config = MmaConfig.m16n8k16_f16_f16()
    block_m, block_n, block_k = 128, 128, 16
    warp_m, warp_n, warp_k = 64, 64, 16
    warp_count_m, warp_count_n, warp_count_k = 2, 2, 1
    mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
    mma_count_m, mma_count_n, mma_count = 4, 8, 1
    threads = warp_count_m * warp_count_n * warp_count_k * 32
    stages = 4

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
        def load_smem_a(
                k0: int,
                a: f16[bs, m_size, k_size],
                smem_a: f16[block_m, block_k]
        ):
            offset_m, offset_n, offset_k = blockIdx.x * block_m, blockIdx.y * block_n, k0 * block_k
            gmem_a = a[blockIdx.z, offset_m:, offset_k:]
            for i, k_seg in repeat(2, 1).spatial(64, 2).on(threadIdx.x):
                k = k_seg * 8
                src_size = 0 if (offset_m + i >= m_size or offset_k + k >= k_size) else min(k_size - (offset_k + k), 8)
                cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=16, src_size=src_size * 2)
            # cp_async_wait_all()

        @hidet.script
        def load_smem_b(
                k0: int,
                b: f16[bs, k_size, n_size],
                smem_b: f16[block_k, block_m]
        ):
            offset_m, offset_n, offset_k = blockIdx.x * block_m, blockIdx.y * block_n, k0 * block_k
            gmem_b = b[blockIdx.z, offset_k:, offset_n:]
            for k, j_seg in repeat(2, 1).spatial(8, 16).on(threadIdx.x):
                j = j_seg * 8
                src_size = 0 if (offset_k + k >= k_size or offset_n + j >= n_size) else min(n_size - (offset_n + j), 8)
                cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=16, src_size=src_size * 2)

        assert 2 <= stages <= 4

        @hidet.script
        def gemm_mma_cp_async_multi_stage_grid(a: f16[bs, m_size, k_size], b: f16[bs, k_size, n_size], c: f16[bs, m_size, n_size]):
            # matrix multiplication, using mma instruction
            attr.cuda_grid_dim = (m_size + block_m - 1) // block_m, (n_size + block_n - 1) // block_n, bs
            attr.cuda_block_dim = threads
            smem_a = tensor('shared', 'float16', [stages, block_m, block_k])
            smem_b = tensor('shared', 'float16', [stages, block_k, block_n])
            regs_a = tensor('register', 'float16', [4, mma_config.a_elements])
            regs_b = tensor('register', 'float16', [8, mma_config.b_elements])
            regs_c = tensor('register', 'float16', [4, 8, mma_config.c_elements])

            for i, j, p in repeat(4, 8, mma_config.c_elements).on(0):
                regs_c[i, j, p] = 0.0

            for stage in range(stages - 1):
                load_smem_a(stage, a, ~smem_a[stage, 0, 0])
                load_smem_b(stage, b, ~smem_b[stage, 0, 0])
                cp_async_commit_group()

            for k0 in range((k_size + block_k - 1) // block_k):
                load_smem_a(k0 + stages - 1, a, ~smem_a[(k0 + stages - 1) % stages, 0, 0])
                load_smem_b(k0 + stages - 1, b, ~smem_b[(k0 + stages - 1) % stages, 0, 0])
                cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 1)
                syncthreads()
                load_regs_a(~smem_a[k0 % stages, 0, 0], regs_a)
                load_regs_b(~smem_b[k0 % stages, 0, 0], regs_b)
                warp_mma(regs_a, regs_b, regs_c)
                syncthreads()
            store_c(regs_c, c)

    ir_module = module.ir_module()
    func = hidet.driver.build_ir_module(ir_module, func_name='gemm_mma_cp_async_multi_stage')
    return func


if __name__ == '__main__':
    # bs, m, n, k = 11, 111, 222, 333
    # bs, m, n, k = 1, 10, 10, 10
    bs, m, n, k = 1, 1280, 1280, 1280
    func = gemm_mma_fp16_cp_async_multi_stage_kernel(bs, m, n, k)
    # a = hidet.randint(0, 2, [bs, m, k], 'float16')
    # b = hidet.randint(0, 2, [bs, k, n], 'float16')
    a = hidet.ones([bs, m, k], 'float16')
    b = hidet.ones([bs, k, n], 'float16')
    c = hidet.zeros([bs, m, n], 'float16')
    func(a, b, c)
    c2 = hidet.ops.matmul(a, b)
    # print(a)
    # print(b)
    # print(c)
    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.numpy(), desired=c2.numpy())
