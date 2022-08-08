import hidet
import numpy as np


def matmul_naive(m_size: int, n_size: int, k_size: int):
    from hidet.ir import IRModule
    from hidet.lang import f32, script, attr
    from hidet.lang.cuda import threadIdx, blockIdx, blockDim

    @script
    def matmul_grid(a: f32[m_size, k_size], b: f32[k_size, n_size], c: f32[m_size, n_size]):
        attr.cuda_block_dim = 16, 16
        attr.cuda_grid_dim = (m_size + 15) / 16, (n_size + 15) / 16
        i = blockIdx.x * blockDim.x + threadIdx.x
        j = blockIdx.y * blockDim.y + threadIdx.y
        if i < m_size and j < n_size:
            c[i, j] = 0.0
            for k in range(k_size):
                c[i, j] += a[i, k] * b[k, j]

    print(matmul_grid)
    return IRModule(funcs={matmul_grid.name: matmul_grid})


def matmul_block(
        m_size: int,
        n_size: int,
        k_size: int,
):
    from hidet.ir import IRModule
    from hidet.lang import f32, attr, tensor, script
    from hidet.lang.cuda import threadIdx, blockIdx, syncthreads
    block_m: int = 128
    block_n: int = 128
    block_k: int = 8
    warp_m: int = 32
    warp_n: int = 64
    inst_m: int = 8
    inst_n: int = 8
    warp_count = block_m // warp_m, block_n // warp_n   # (4, 2)
    threads = warp_count[0] * warp_count[1] * 32        # 256

    @script
    def matmul_grid(a: f32[m_size, k_size], b: f32[k_size, n_size], c: f32[m_size, n_size]):
        attr.cuda_block_dim = threads
        attr.cuda_grid_dim = (m_size + block_m - 1) / block_m, (n_size + block_n - 1) / block_n
        smem_a = tensor('shared', 'float32', [block_m, block_k])
        smem_b = tensor('shared', 'float32', [block_k, block_n])
        regs_c = tensor('register', 'float32', [inst_m, inst_n])
        warp_id = threadIdx.x / 32
        lane_id = threadIdx.x % 32
        warp_i, warp_j = warp_id / 2, warp_id % 2
        thread_i, thread_j = lane_id / 8, lane_id % 8
        for i in range(inst_m):
            for j in range(inst_n):
                regs_c[i, j] = 0.0
        for k0 in range((k_size + block_k - 1) // block_k):
            # load A and B from global memory to shared memory
            for i1 in range(4):
                i = i1 * 32 + threadIdx.x / block_k
                k = threadIdx.x % block_k
                gi, gk = blockIdx.x * block_m + i, k0 * block_k + k
                smem_a[i, k] = a[gi, gk] if gi < m_size and gk < k_size else 0.0
            for k1 in range(4):
                k = k1 * 2 + threadIdx.x / block_n
                j = threadIdx.x % block_n
                gk, gj = k0 * block_k + k, blockIdx.y * block_n + j
                smem_b[k, j] = b[gk, gj] if gk < k_size and gj < n_size else 0.0
            syncthreads()
            # matrix multiply accumulate
            for i1 in range(inst_m):
                for j1 in range(inst_n):
                    for k in range(block_k):
                        i = warp_i * warp_m + i1 * 4 + thread_i
                        j = warp_j * warp_n + j1 * 8 + thread_j
                        regs_c[i1, j1] += smem_a[i, k] * smem_b[k, j]
            syncthreads()
        for i1 in range(inst_m):
            for j1 in range(inst_n):
                i = warp_i * warp_m + i1 * 4 + thread_i
                j = warp_j * warp_n + j1 * 8 + thread_j
                gi, gj = blockIdx.x * block_m + i, blockIdx.y * block_n + j
                if gi < m_size and gj < n_size:
                    c[gi, gj] = regs_c[i1, j1]

    print(matmul_grid)
    return IRModule(funcs={matmul_grid.name: matmul_grid})


def matmul_use_mapping(m_size: int, n_size: int, k_size: int):
    from hidet.ir import IRModule, Var
    from hidet.lang import script, f32, tensor, attr, printf
    from hidet.lang.mapping import spatial, repeat, chain
    from hidet.lang.layout import row_layout, col_layout, local_layout
    from hidet.lang.cuda import threadIdx, blockIdx, syncthreads
    block_m: int = 128
    block_n: int = 128
    block_k: int = 8
    warp_m: int = 32
    warp_n: int = 64
    warp_count = block_m // warp_m, block_n // warp_n   # (4, 2)
    threads = warp_count[0] * warp_count[1] * 32        # 256

    @script
    def matmul_grid(a: f32[m_size, k_size], b: f32[k_size, n_size], c: f32[m_size, n_size]):
        attr.cuda_block_dim = threads
        attr.cuda_grid_dim = (m_size + block_m - 1) / block_m, (n_size + block_n - 1) / block_n
        offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
        smem_a = tensor('shared', 'float32', [block_m, block_k])
        smem_b = tensor('shared', 'float32', [block_k, block_n])
        regs_c = tensor('register', 'float32', layout=local_layout(4, 2) * row_layout(8, 8) * local_layout(4, 8))
        gmem_c = c[offset_m:, offset_n:]

        c_mapping = chain(spatial(4, 2), repeat(8, 8), spatial(4, 8))
        for i, j in c_mapping.on(threadIdx.x):
            regs_c[i, j] = 0.0
        for k0 in range((k_size + block_k - 1) // block_k):
            gmem_a = a[offset_m:, k0 * block_k:]
            gmem_b = b[k0 * block_k:, offset_n:]
            for i, k in chain(repeat(4, 1), spatial(32, 8)).on(threadIdx.x):
                smem_a[i, k] = gmem_a.read([i, k], protected=True)
            for k, j in chain(repeat(4, 1), spatial(2, 128)).on(threadIdx.x):
                smem_b[k, j] = gmem_b.read([k, j], protected=True)
            syncthreads()
            for i, j in c_mapping.on(threadIdx.x):
                for k in range(block_k):
                    regs_c[i, j] += smem_a[i, k] * smem_b[k, j]
            syncthreads()
        for i, j in c_mapping.on(threadIdx.x):
            gmem_c.write([i, j], regs_c[i, j], protected=True)

    print(matmul_grid)
    return IRModule(funcs={matmul_grid.name: matmul_grid})


def main():
    # m_size, n_size, k_size = 1024, 1024, 1024
    m_size, n_size, k_size = 127, 127, 127
    # ir_module = matmul_naive(m_size, n_size, k_size)
    # ir_module = matmul_block(m_size, n_size, k_size)
    ir_module = matmul_use_mapping(m_size, n_size, k_size)
    func = hidet.driver.build_ir_module(ir_module, func_name='matmul')
    a = hidet.randint(2, shape=[m_size, k_size], dtype='float32')
    b = hidet.randint(2, shape=[k_size, n_size], dtype='float32')
    # a = hidet.ones([m_size, k_size])
    # b = hidet.ones([k_size, n_size])
    c = hidet.zeros([m_size, n_size])
    func(a, b, c)
    # print(func.profile(a, b, c))
    hidet.utils.cuda.device_synchronize()
    c2 = hidet.ops.matmul(a, b)
    # with hidet.utils.Timer('normal') as timer:
    #     for t in range(100):
    #         c2 = hidet.ops.matmul(a, b)
    #     hidet.utils.cuda.device_synchronize()
    # print(timer.elapsed_seconds())
    np.testing.assert_allclose(actual=c.numpy(), desired=c2.numpy())


if __name__ == '__main__':
    # import astunparse
    #     print(ast.dump(ast.parse("""
    # if a < 5:
    #     pass
    #     """)))
    main()
