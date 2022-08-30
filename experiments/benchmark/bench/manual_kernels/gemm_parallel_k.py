from hidet.runtime import CompiledFunction
import numpy as np
import hidet
from bench.common import BenchResult, benchmark_run


def gemm_mma_fp16_parallel_k_kernel(batch_size, m_size, n_size, k_size) -> CompiledFunction:
    from hidet.lang import f16, f32, spatial, repeat, tensor, attr, printf, cast, col_spatial, view, u32, PointerType, tensor_pointer, grid, i32, void_p, var_of_function, static, void
    from hidet.lang.layout import row_layout, col_layout, local_layout, DataLayout
    from hidet.lang.mapping import repeat, spatial
    from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory
    from hidet.lang.cuda import MmaConfig, mma_sync, cp_async, cp_async_wait_all, cvta_generic_to_shared, ldmatrix, cp_async_wait_group, cp_async_commit_group, acquire_lock, release_lock
    from hidet.lang.cuda import acquire_seq_semaphore, release_seq_semaphore
    from hidet.lang.runtime import request_workspace

    # optimize for 128x768x3072
    # mma_config = MmaConfig.m16n8k16_f16_f16()
    # block_m, block_n, block_k = 128, 128, 64
    # warp_m, warp_n, warp_k = 64, 32, 64
    # mma_config = MmaConfig.m16n8k8_f16_f16()
    mma_config = MmaConfig.m16n8k16_f16_f16()
    block_m, block_n, block_k = 64, 128, 16
    warp_m, warp_n, warp_k = 32, 64, 16
    split_k_size = 256

    # block_m, block_n, block_k = 128, 128, 16
    # warp_m, warp_n, warp_k = 64, 64, 16
    block_count_m, block_count_n, block_count_k = (m_size + block_m - 1) // block_m, (n_size + block_n - 1) // block_n, (k_size + split_k_size - 1) // split_k_size
    warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
    mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
    mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
    threads = warp_count_m * warp_count_n * warp_count_k * 32
    assert block_m % warp_m == block_n % warp_n == block_k % warp_k == 0
    assert warp_m % mma_m == warp_n % mma_n == warp_k % mma_k == 0
    assert block_n % 64 == 0
    smem_a_type = tensor(
        'shared', 'float16', shape=[block_m, block_k],
        layout=row_layout(block_m, block_k // 8).swizzle(1) * row_layout(1, 8)
    )
    smem_b_type = tensor(
        'shared', 'float16', shape=[block_k, block_n],
        layout=row_layout(block_k // 8, block_n // 64) * row_layout(8, 8).swizzle(1) * row_layout(1, 8)
    )
    load_smem_a_map = repeat(block_m // (threads // (block_k // 8)), 1).spatial(threads // (block_k // 8), block_k // 8)
    load_smem_b_map = repeat(block_k // (threads // (block_n // 8)), 1).spatial(threads // (block_n // 8), block_n // 8)

    with hidet.script_module() as module:
        @hidet.script
        def load_regs_a(
                mi: int,
                k1: int,
                smem_a: smem_a_type,
                regs_a: tensor('register', 'float16', [mma_config.a_elements])
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
                regs_b: tensor('register', 'float16', [mma_config.b_elements])
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
                regs_a: tensor('register', 'float16', [mma_config.a_elements]),
                regs_b: tensor('register', 'float16', [mma_config.b_elements]),
                regs_c: tensor('register', 'float16', [mma_config.c_elements])
        ):
            mma_sync(mma_config, regs_a, regs_b, regs_c)

        assert warp_count_k == 1, "current store_c implementation only supports warp_count_k == 1"

        def inbound(indices, bounds):
            from hidet.ir.expr import And, LessThan
            return And.join_list([LessThan(index, bound) for index, bound in zip(indices, bounds)])

        @hidet.script
        def store_c(
                regs_c: tensor('register', 'float16', [mma_count_m, mma_count_n, mma_config.c_elements]),
                c: tensor('global', 'float16', [batch_size, m_size, n_size]),
                locks: tensor('global', 'int32', [batch_size, block_count_m, block_count_n])
        ):
            split_k_tile, batch = spatial(block_count_k, batch_size).single_task_of(blockIdx.z)
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
            # gmem_c = c[blockIdx.z, offset_m:, offset_n:]
            lock = ~locks[batch, blockIdx.x, blockIdx.y]

            acquire_seq_semaphore(lock, split_k_tile)
            if split_k_tile == 0:
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    for mi, mj in repeat(mma_count_m, mma_count_n).on(0):
                        p = 0
                        for ii, jj in mma_config.c_store_map.on(lane_id):
                            b, i, j = [
                                batch,
                                offset_m + (wi * warp_m + mi * mma_m + ii),
                                offset_n + (wj * warp_n + mj * mma_n + jj)
                            ]
                            if inbound([i, j], [m_size, n_size]):
                                c[b, i, j] = regs_c[mi, mj, p]
                            p += 1
            else:
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    for mi, mj in repeat(mma_count_m, mma_count_n).on(0):
                        p = 0
                        for ii, jj in mma_config.c_store_map.on(lane_id):
                            b, i, j = [
                                batch,
                                offset_m + (wi * warp_m + mi * mma_m + ii),
                                offset_n + (wj * warp_n + mj * mma_n + jj)
                            ]
                            if inbound([i, j], [m_size, n_size]):
                                c[b, i, j] = c[b, i, j] + regs_c[mi, mj, p]
                            p += 1
            release_seq_semaphore(lock, (split_k_tile + 1) % block_count_k)

        @hidet.script
        def load_smem_a(
                k0: int,
                a: f16[batch_size, m_size, k_size],
                smem_a: smem_a_type
        ):
            split_k_tile, batch = spatial(block_count_k, batch_size).single_task_of(blockIdx.z)
            offset_m, offset_n, offset_k = blockIdx.x * block_m, blockIdx.y * block_n, k0 * block_k + split_k_tile * split_k_size
            gmem_a = a[batch, offset_m:, offset_k:]
            for i, k_seg in load_smem_a_map.on(threadIdx.x):
                k = k_seg * 8
                src_size = 0 if (offset_m + i >= m_size or offset_k + k >= k_size) else min(k_size - (offset_k + k), 8)
                cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=16, src_size=src_size * 2, cache_level='global')

        @hidet.script
        def load_smem_b(
                k0: int,
                b: f16[batch_size, k_size, n_size],
                smem_b: smem_b_type
        ):
            split_k_tile, batch = spatial(block_count_k, batch_size).single_task_of(blockIdx.z)
            offset_m, offset_n, offset_k = blockIdx.x * block_m, blockIdx.y * block_n, k0 * block_k + split_k_tile * split_k_size
            gmem_b = b[batch, offset_k:, offset_n:]
            for k, j_seg in load_smem_b_map.on(threadIdx.x):
                j = j_seg * 8
                src_size = 0 if (offset_k + k >= k_size or offset_n + j >= n_size) else min(n_size - (offset_n + j), 8)
                cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=16, src_size=src_size * 2, cache_level='global')

        @hidet.script
        def gemm_mma_fp16_parallel_k_grid(
                a: f16[batch_size, m_size, k_size],
                b: f16[batch_size, k_size, n_size],
                c: f16[batch_size, m_size, n_size],
                locks: i32[batch_size, block_count_m, block_count_n]
        ):
            # matrix multiplication, using mma instruction
            attr.cuda_grid_dim = (m_size + block_m - 1) // block_m, (n_size + block_n - 1) // block_n, batch_size * block_count_k
            attr.cuda_block_dim = threads
            attr.cuda_dynamic_smem_bytes = 2 * (block_m + block_n) * block_k * 2  # the second 2 means '2 bytes per float16'
            smem_a: tensor_pointer('shared', 'float16', shape=[2, block_m, block_k], layout=DataLayout.concat(row_layout(2), smem_a_type.layout))
            smem_b: tensor_pointer('shared', 'float16', shape=[2, block_k, block_n], layout=DataLayout.concat(row_layout(2), smem_b_type.layout))
            smem_a = dynamic_shared_memory(byte_offset=0, dtype=f16)
            smem_b = dynamic_shared_memory(byte_offset=2 * block_m * block_k * 2, dtype=f16)
            regs_a = tensor('register', 'float16', [2, mma_count_m, mma_config.a_elements])
            regs_b = tensor('register', 'float16', [2, mma_count_n, mma_config.b_elements])
            regs_c = tensor('register', 'float16', [mma_count_m, mma_count_n, mma_config.c_elements])

            for i, j, p in grid(mma_count_m, mma_count_n, mma_config.c_elements):
                regs_c[i, j, p] = 0.0

            load_smem_a(0, a, ~smem_a[0, 0, 0])
            load_smem_b(0, b, ~smem_b[0, 0, 0])
            cp_async_wait_all()
            syncthreads()
            for k0 in range((split_k_size + block_k - 1) // block_k):
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
            store_c(regs_c, c, locks)

        from hidet.lang.runtime import request_workspace

        @hidet.script
        def gemm_mma_fp16_parallel_k(
                num_args: int,
                arg_types: ~i32,
                args: ~void_p
        ):
            attr.func_kind = 'packed_func'
            attr.packed_func = var_of_function(gemm_mma_fp16_parallel_k_grid)
            locks = static(tensor_pointer('global', dtype='int32', shape=[batch_size, block_count_m, block_count_n]))
            if locks == 0:
                locks = request_workspace(nbytes=4 * batch_size * block_count_m * block_count_n, require_clean=True)
            assert num_args == 3
            assert arg_types[0] == 3
            assert arg_types[1] == 3
            assert arg_types[2] == 3
            gemm_mma_fp16_parallel_k_grid(
                cast(args[0], ~f16),
                cast(args[1], ~f16),
                cast(args[2], ~f16),
                locks
            )

    ir_module = module.ir_module()
    func = hidet.driver.build_ir_module(ir_module, func_name='gemm_mma_fp16_parallel_k', func_type=hidet.ir.FuncType([~f32, ~f32, ~f32], void))
    return func


if __name__ == '__main__':
    from hidet.lang import f16

    # bs, m, n, k = 11, 111, 222, 333
    # bs, m, n, k = 1, 10, 10, 10
    bs, m, n, k = 1, 1280, 1280, 1280
    func = gemm_mma_fp16_parallel_k_kernel(bs, m, n, k)
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
