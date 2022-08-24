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
    warp_count = block_m // warp_m, block_n // warp_n  # (4, 2)
    threads = warp_count[0] * warp_count[1] * 32  # 256

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
    warp_count = block_m // warp_m, block_n // warp_n  # (4, 2)
    threads = warp_count[0] * warp_count[1] * 32  # 256

    @script
    def matmul_grid(a: f32[m_size, k_size], b: f32[k_size, n_size], c: f32[m_size, n_size]):
        attr.cuda_block_dim = threads
        attr.cuda_grid_dim = (m_size + block_m - 1) / block_m, (n_size + block_n - 1) / block_n
        offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
        smem_a = tensor('shared', 'float32', [block_m, block_k])
        smem_b = tensor('shared', 'float32', [block_k, block_n])
        regs_c = tensor('register', 'float32', layout=local_layout(4, 2) * row_layout(8, 8) * local_layout(4, 8))
        gmem_c = c[offset_m:, offset_n:]

        c_mapping = spatial(4, 2).repeat(8, 8).spatial(4, 8)
        for i, j in c_mapping.on(threadIdx.x):
            regs_c[i, j] = 0.0
        for k0 in range((k_size + block_k - 1) // block_k):
            gmem_a = a[offset_m:, k0 * block_k:]
            gmem_b = b[k0 * block_k:, offset_n:]
            for i, k in repeat(4, 1).spatial(32, 8).on(threadIdx.x):
                smem_a[i, k] = gmem_a.read([i, k], protected=True)
            for k, j in repeat(4, 1).spatial(2, 128).on(threadIdx.x):
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


def demo_call_example():
    from hidet.ir import IRModule
    from hidet.lang import attr, i32
    from hidet.lang import printf
    from hidet.lang.cuda import threadIdx

    with hidet.script_module() as module:
        @hidet.script
        def print_index(idx: i32):
            printf(r'threadIdx.x %d\n', idx)

        @hidet.script
        def call_example_grid():
            attr.cuda_grid_dim = 1
            attr.cuda_block_dim = 12
            print_index(threadIdx.x)

    func = hidet.driver.build_ir_module(module.ir_module(), func_name='call_example', verbose=True)
    func()


def demo_cvta():
    from hidet.ir import IRModule
    from hidet.lang import attr, i32, tensor
    from hidet.lang import printf
    from hidet.lang.cuda import threadIdx
    from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

    with hidet.script_module() as module:
        @hidet.script
        def cvta_grid():
            attr.cuda_grid_dim = 1
            attr.cuda_block_dim = 1
            smem = tensor('shared', 'float32', shape=[10])
            smem_ptr = cvta_generic_to_shared(~smem[3])
            printf(r'%p %d\n', ~smem[0], smem_ptr)

    func = hidet.driver.build_ir_module(module.ir_module(), func_name='cvta', verbose=True)
    func()


def demo_cp_async():
    from hidet.lang import f32, tensor, attr, printf
    from hidet.lang.cuda import threadIdx, blockIdx, cp_async, cp_async_commit_group, cp_async_wait_all, syncthreads, cvta_generic_to_shared
    with hidet.script_module() as module:
        @hidet.script
        def demo_cp_async_grid(a: f32[4]):
            attr.cuda_grid_dim = 1
            attr.cuda_block_dim = 1
            smem = tensor('shared', 'float32', [4])
            cp_async(smem, a, cp_size=16)
            cp_async_wait_all()
            # syncthreads()
            for i in range(4):
                assert smem[i] == a[i]

    func = hidet.driver.build_ir_module(module.ir_module(), func_name='demo_cp_async', verbose=True, keep_ptx=True)
    a = hidet.array(list(range(4))).to('float32').cuda()
    print(a)
    func(a)


def demo_ldmatrix():
    from hidet.lang import script, f16, f32, tensor, attr, spatial, repeat, printf, cast
    from hidet.lang.cuda import ldmatrix, threadIdx, syncthreads
    from hidet.ir.dialects.lowlevel import PointerType, Dereference, TensorPointerType

    with hidet.script_module() as module:
        @hidet.script
        def demo_ldmatrix_grid(a: f16[8, 8]):
            attr.cuda_grid_dim = 1
            attr.cuda_block_dim = 32

            smem = tensor('shared', 'float16', [8, 8])
            regs = tensor('register', 'uint32', [1])

            for i, j in repeat(2, 1).spatial(4, 8).on(threadIdx.x):
                smem[i, j] = a[i, j]
            syncthreads()
            ldmatrix(regs=[regs[0]], smem_addr=~smem[threadIdx.x][0])
            regs_view = cast(~regs[0], TensorPointerType('register', dtype='float16', shape=[2]))
            printf(r'%d %.0f %.0f\n', threadIdx.x, cast(regs_view[0], f32), cast(regs_view[1], f32))
    func = hidet.driver.build_ir_module(module.ir_module(), func_name='demo_ldmatrix', verbose=True, keep_ptx=True)
    a = hidet.array(np.arange(64).astype(np.float16)).cuda()
    print(a)
    func(a)


def demo_ldmatrix_x4():
    from hidet.lang import script, f16, f32, tensor, attr, spatial, repeat, printf, cast, view, u32, col_spatial
    from hidet.lang.cuda import ldmatrix, threadIdx, syncthreads
    from hidet.ir.dialects.lowlevel import PointerType, Dereference, TensorPointerType

    with hidet.script_module() as module:
        @hidet.script
        def demo_ldmatrix_grid(a: f16[16, 16]):
            attr.cuda_grid_dim = 1
            attr.cuda_block_dim = 32

            smem = tensor('shared', 'float16', [16, 16])
            regs = tensor('register', 'float16', [8])
            # regs = tensor('register', 'uint32', [4])

            for i, j in repeat(8, 1).spatial(2, 16).on(threadIdx.x):
                smem[i, j] = a[i, j]
            syncthreads()
            u32_regs = view(regs, u32[4])
            p, q = col_spatial(16, 2).map(threadIdx.x)
            ldmatrix(regs=[u32_regs[0], u32_regs[1], u32_regs[2], u32_regs[3]], smem_addr=~smem[p, q * 8])
            printf(
                r'threadIdx.x %d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n',
                threadIdx.x,
                cast(regs[0], f32),
                cast(regs[1], f32),
                cast(regs[2], f32),
                cast(regs[3], f32),
                cast(regs[4], f32),
                cast(regs[5], f32),
                cast(regs[6], f32),
                cast(regs[7], f32)
            )
            # regs_view = cast(~regs[0], TensorPointerType('register', dtype='float16', shape=[2]))
            # printf(r'%d %.0f %.0f\n', threadIdx.x, cast(regs_view[0], f32), cast(regs_view[1], f32))
    func = hidet.driver.build_ir_module(module.ir_module(), func_name='demo_ldmatrix', verbose=True, keep_ptx=True)
    a = hidet.array(np.arange(8 * 8 * 4).astype(np.float16)).cuda()
    print(a)
    func(a)


def demo_ldmatrix_x4_trans():
    from hidet.lang import script, f16, f32, tensor, attr, spatial, repeat, printf, cast, view, u32, col_spatial
    from hidet.lang.cuda import ldmatrix, threadIdx, syncthreads
    from hidet.ir.dialects.lowlevel import PointerType, Dereference, TensorPointerType

    with hidet.script_module() as module:
        @hidet.script
        def demo_ldmatrix_grid(a: f16[16, 16]):
            attr.cuda_grid_dim = 1
            attr.cuda_block_dim = 32

            smem = tensor('shared', 'float16', [16, 24])
            regs = tensor('register', 'float16', [8])
            # regs = tensor('register', 'uint32', [4])

            for i, j in repeat(8, 1).spatial(2, 16).on(threadIdx.x):
                smem[i, j] = a[i, j]
            syncthreads()
            u32_regs = view(regs, u32[4])
            p, q = col_spatial(16, 2).map(threadIdx.x)
            ldmatrix(regs=[u32_regs[0], u32_regs[1], u32_regs[2], u32_regs[3]], smem_addr=~smem[p, q * 8], trans=True)
            printf(
                r'threadIdx.x %2d: %3.0f %3.0f %3.0f %3.0f %3.0f %3.0f %3.0f %3.0f\n',
                threadIdx.x,
                cast(regs[0], f32),
                cast(regs[1], f32),
                cast(regs[2], f32),
                cast(regs[3], f32),
                cast(regs[4], f32),
                cast(regs[5], f32),
                cast(regs[6], f32),
                cast(regs[7], f32)
            )
            # regs_view = cast(~regs[0], TensorPointerType('register', dtype='float16', shape=[2]))
            # printf(r'%d %.0f %.0f\n', threadIdx.x, cast(regs_view[0], f32), cast(regs_view[1], f32))
    func = hidet.driver.build_ir_module(module.ir_module(), func_name='demo_ldmatrix', verbose=True, keep_ptx=True)
    a = hidet.array(np.arange(8 * 8 * 4).astype(np.float16)).cuda()
    print(a)
    func(a)


def demo_cp_async_bank_conflicts():
    from hidet.lang import script, f16, f32, tensor, attr, spatial, repeat, printf, cast, view, u32, col_spatial, i64, i32
    from hidet.lang.cuda import ldmatrix, threadIdx, syncthreads, cp_async, cp_async_wait_all, cvta_generic_to_shared
    from hidet.ir.dialects.lowlevel import PointerType, Dereference, TensorPointerType

    size = 256
    with hidet.script_module() as module:
        @hidet.script
        def demo_cp_async_bank_conflicts_grid(a: f16[size]):
            attr.cuda_grid_dim = 1
            attr.cuda_block_dim = 32

            smem = tensor('shared', 'float16', [size])
            group = threadIdx.x / 4
            index_in_group = threadIdx.x % 4
            idx = index_in_group * 8 + group
            idx = idx ^ 23
            # printf(r'idx %d threadIdx.x %d\n', idx, threadIdx.x)
            # idx = threadIdx.x ^ 1
            cp_async(
                dst=~smem[idx * 8],
                src=~a[(idx ^ 13) * 8],
                cp_size=16,
                cache_level='global'
            )
            cp_async_wait_all()
            printf(r'threadIdx %d \t gmem_addr %d \t smem_addr %d\n', threadIdx.x, cast(cast(~a[(idx ^ 13) * 8], i64) - cast(~a[0], i64), i32), cvta_generic_to_shared(~smem[idx * 8]))
            # printf(r'threadIdx.x %d  %.0f %.0f\n', threadIdx.x, cast(smem[threadIdx.x * 2], f32), cast(smem[threadIdx.x * 2 + 1], f32))

            # regs_view = cast(~regs[0], TensorPointerType('register', dtype='float16', shape=[2]))
            # printf(r'%d %.0f %.0f\n', threadIdx.x, cast(regs_view[0], f32), cast(regs_view[1], f32))
    func = hidet.driver.build_ir_module(module.ir_module(), func_name='demo_cp_async_bank_conflicts', verbose=True, keep_ptx=True)
    a = hidet.array(np.arange(size).astype(np.float16)).cuda()
    print(a)
    func(a)


def demo_cp_async_bank_conflicts_v2():
    from hidet.lang import script, f16, f32, tensor, attr, spatial, repeat, printf, cast, view, u32, col_spatial
    from hidet.lang.cuda import ldmatrix, threadIdx, syncthreads, cp_async, cp_async_wait_all
    from hidet.lang.layout import row_layout, col_layout
    from hidet.ir.dialects.lowlevel import PointerType, Dereference, TensorPointerType

    print(row_layout(2, 1) * (row_layout(2, 2).swizzle(1)) * row_layout(4, 8))

    with hidet.script_module() as module:
        @hidet.script
        def demo_cp_async_bank_conflicts_grid(a: f16[16, 16], b: f16[16, 16]):
            attr.cuda_grid_dim = 1
            attr.cuda_block_dim = 32

            smem = tensor('shared', 'float16', [16, 16], layout=row_layout(2, 1) * (row_layout(2, 2).swizzle(1)) * row_layout(4, 8))
            for i, j in spatial(16, 2).on(threadIdx.x):
                cp_async(
                    dst=~smem[i, j * 8],
                    src=~a[i, j * 8],
                    cp_size=16,
                )
            cp_async_wait_all()
            syncthreads()
            for i, j in repeat(8, 1).spatial(2, 16).on(threadIdx.x):
                b[i, j] = smem[i, j]
            # printf(r'threadIdx.x %d  %.0f %.0f\n', threadIdx.x, cast(smem[threadIdx.x * 2], f32), cast(smem[threadIdx.x * 2 + 1], f32))
            # regs_view = cast(~regs[0], TensorPointerType('register', dtype='float16', shape=[2]))
            # printf(r'%d %.0f %.0f\n', threadIdx.x, cast(regs_view[0], f32), cast(regs_view[1], f32))
    func = hidet.driver.build_ir_module(module.ir_module(), func_name='demo_cp_async_bank_conflicts', verbose=True, keep_ptx=True)
    a = hidet.array(np.arange(16 * 16).astype(np.float16)).cuda()
    b = hidet.array(np.zeros(16 * 16).astype(np.float16)).cuda()
    # print(a)
    func(a, b)
    print(b)


def demo_cp_async_ldmatrix_bank_conflicts():
    from hidet.lang import script, f16, f32, tensor, attr, spatial, repeat, printf, cast, view, u32, col_spatial, i64, i32
    from hidet.lang.cuda import ldmatrix, threadIdx, syncthreads, cp_async, cp_async_wait_all, blockIdx, cvta_generic_to_shared
    from hidet.lang.layout import row_layout, col_layout
    from hidet.ir.dialects.lowlevel import PointerType, Dereference, TensorPointerType

    block_count_m, block_count_n = 1, 1
    block_m, block_n = 128, 64
    smem_layout = row_layout(block_m // 16, block_n // 16) * row_layout(16, 2).swizzle(1, log_step=2) * row_layout(1, 8)
    # smem_layout = row_layout(block_m, block_n)
    with hidet.script_module() as module:
        @hidet.script
        def demo_cp_async_ldmatrix_bank_conflicts_grid(
                a: f16[block_count_m * block_m, block_count_n * block_n]
        ):
            attr.cuda_grid_dim = block_count_m, block_count_n
            # attr.cuda_block_dim = 256
            attr.cuda_block_dim = 32
            offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n

            smem = tensor('shared', 'float16', [block_m, block_n], layout=smem_layout)
            # for i, j in repeat(1, 1).spatial(32, 8).on(threadIdx.x):
            for i, j in repeat(1, 1).spatial(32, 8).on(threadIdx.x):
                cp_async(
                    dst=~smem[i, j * 8],
                    src=~a[offset_m + i, offset_n + j * 8],
                    cp_size=16,
                    cache_level='global'
                )
                printf(
                    r'threadIdx %d \t gmem_addr %d \t smem_addr %d\n',
                    threadIdx.x,
                    cast(cast(~a[offset_m + i, offset_n + j * 8], i64) - cast(~a[0, 0], i64), i32),
                    # cvta_generic_to_shared(~smem[smem_idx])
                    cvta_generic_to_shared(~smem[i, j * 8])
                    # cvta_generic_to_shared(smem_addr)
                )
                # printf(r'threadIdx %d smem_addr %d\n', threadIdx.x, cvta_generic_to_shared(~smem[i, j * 8]))
            cp_async_wait_all()
            syncthreads()
    func = hidet.driver.build_ir_module(module.ir_module(), func_name='demo_cp_async_ldmatrix_bank_conflicts', verbose=True, keep_ptx=True)
    a = hidet.array(np.arange(block_m * block_n * block_count_m * block_count_n).astype(np.float16)).cuda()
    # print(a)
    func(a)
    # print(b)


def demo_cp_async_ldmatrix_bank_conflicts_16x128():
    from hidet.lang import script, f16, f32, tensor, attr, spatial, repeat, printf, cast, view, u32, col_spatial, i64, i32
    from hidet.lang.cuda import ldmatrix, threadIdx, syncthreads, cp_async, cp_async_wait_all, blockIdx, cvta_generic_to_shared
    from hidet.lang.layout import row_layout, col_layout
    from hidet.ir.dialects.lowlevel import PointerType, Dereference, TensorPointerType

    # print((row_layout(16, 2).swizzle(1, log_step=3)) * row_layout(1, 8))

    block_m, block_n = 2, 2
    with hidet.script_module() as module:
        @hidet.script
        def demo_cp_async_ldmatrix_bank_conflicts_grid(
                a: f16[16 * block_m, 128 * block_n]
        ):
            attr.cuda_grid_dim = block_m, block_n
            attr.cuda_block_dim = 128
            offset_m, offset_n = blockIdx.x * 16, blockIdx.y * 128

            smem = tensor('shared', 'float16', [16, 128], layout=row_layout(2, 2) * row_layout(8, 8).swizzle(1) * row_layout(1, 8))
            # smem = tensor('shared', 'float16', [16, 128], layout=row_layout(16, 128))
            for i, j in repeat(2, 1).spatial(8, 16).on(threadIdx.x):
                cp_async(
                    dst=~smem[i, j * 8],
                    src=~a[offset_m + i, offset_n + j * 8],
                    cp_size=16,
                    cache_level='global'
                )
                # printf(
                #     r'threadIdx %d \t gmem_addr %d \t smem_addr %d\n',
                #     threadIdx.x,
                #     cast(cast(~a[offset_m + i, offset_n + j * 8], i64) - cast(~a[0, 0], i64), i32),
                #     # cvta_generic_to_shared(~smem[smem_idx])
                #     cvta_generic_to_shared(~smem[i, j * 8])
                #     # cvta_generic_to_shared(smem_addr)
                # )
                # printf(r'threadIdx %d smem_addr %d\n', threadIdx.x, cvta_generic_to_shared(~smem[i, j * 8]))
            cp_async_wait_all()
            syncthreads()
    func = hidet.driver.build_ir_module(module.ir_module(), func_name='demo_cp_async_ldmatrix_bank_conflicts', verbose=True, keep_ptx=True)
    a = hidet.array(np.arange(128 * 16 * block_m * block_n).astype(np.float16)).cuda()
    b = hidet.array(np.zeros(128 * 16 * block_m * block_n).astype(np.float16)).cuda()
    # print(a)
    func(a)
    # print(b)


def demo_for_grid():
    from hidet.lang import grid, printf
    with hidet.script_module() as module:
        @hidet.script
        def func():
            for i, j in grid(3, 4):
                printf(r'%d %d\n', i, j)
    print(module.ir_module())


def demo_while_grid():
    from hidet.lang import printf, attr
    with hidet.script_module() as module:
        @hidet.script
        def func_grid():
            attr.cuda_grid_dim = 1
            attr.cuda_block_dim = 1
            t = 0
            while t < 5:
                if t == 1:
                    t += 1
                    continue
                t += 1
                if t == 3:
                    break
                printf(r'hi %d\n', t)
    print(module.ir_module())
    func = hidet.driver.build_ir_module(module.ir_module(), func_name='func', verbose=True, keep_ptx=True)
    func()


def demo_load_store():
    from hidet.lang import printf, attr, i32
    from hidet.lang.cuda import load, store, threadIdx

    with hidet.script_module() as module:
        @hidet.script
        def func_grid(a: i32[5]):
            attr.cuda_grid_dim = 1
            attr.cuda_block_dim = 1

            b = load(~a[1])
            if threadIdx.x == 0:
                printf(r'%d\n', b)
    print(module.ir_module())
    func = hidet.driver.build_ir_module(module.ir_module(), func_name='func', verbose=True, keep_ptx=True)
    a = hidet.array(np.arange(5, dtype=np.int32)).cuda()
    func(a)


def demo_mutex():
    from hidet.lang import printf, attr, i32
    from hidet.lang.cuda import load, store, nano_sleep, atomic_cas, blockIdx, threadIdx, shfl_sync, syncthreads_and, syncthreads, acquire_lock, release_lock

    with hidet.script_module() as module:
        @hidet.script
        def func_grid(mutex_lock: i32[1]):
            attr.cuda_grid_dim = 5
            attr.cuda_block_dim = 32
            # status: i32 = 1
            # while syncthreads_and(status == 1):
            #     if threadIdx.x == 0:
            #         status = atomic_cas(mutex_lock, 0, 1)
            # got the lock, begin of critical region
            acquire_lock(mutex_lock)
            if threadIdx.x == 0:
                printf(r'blockIdx.x %d\n', blockIdx.x)
                printf(r'start\n')
                nano_sleep(4000000000)  # sleep 1 seconds
                printf(r'end\n')
            release_lock(mutex_lock)
            # end of critical region
            # syncthreads()
            # if threadIdx.x == 0:
            #     status = atomic_cas(mutex_lock, 1, 0)
    func = hidet.driver.build_ir_module(module.ir_module(), func_name='func', verbose=True, keep_ptx=True)
    a = hidet.zeros([1], dtype='int32')
    func(a)
    print(a)


if __name__ == '__main__':
    # main()
    # demo_call_example()
    demo_cvta()
    # demo_cp_async()
    # demo_ldmatrix()
    # demo_ldmatrix_x4()
    # demo_ldmatrix_x4_trans()
    # demo_cp_async_bank_conflicts()
    # demo_cp_async_bank_conflicts_v2()
    # demo_cp_async_ldmatrix_bank_conflicts()
    # demo_cp_async_ldmatrix_bank_conflicts_16x128()
    # demo_for_grid()
    # demo_while_grid()
    # demo_load_store()
    # demo_mutex()
