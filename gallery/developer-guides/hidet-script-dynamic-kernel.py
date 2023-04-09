"""
Writing Dynamic kernel
======================

.. todo::

  More details about hidet script and how to write dynamic kernel are coming soon.

"""
import numpy.testing
import hidet


def matmul_simt_kernel():
    from hidet.lang import attr
    from hidet.lang import float32, int32
    from hidet.lang import as_tensor_pointer, tensor
    from hidet.lang.cuda import threadIdx, blockIdx, syncthreads
    from hidet.lang.mapping import repeat, spatial, auto_map
    from hidet.lang.layout import row_layout, local_layout

    warps_m, warps_n = 4, 2  # we use 4x2 warps
    warp_m, warp_n = 2, 2  # each warp repeats 2x2 times
    warp_map_m, warp_map_n = 2, 16  # each warp has 2x16 threads
    thread_m, thread_n = 4, 4  # each thread repeats 4x4 times

    # block_size = (64, 256, 8)
    block_m_size, block_n_size = (
        warps_m * warp_m * warp_map_m * thread_m,
        warps_n * warp_n * warp_map_n * thread_n,
    )
    block_k_size = 8
    num_warps = warps_m * warps_n  # 8
    num_threads = num_warps * 32  # 256

    with hidet.lang.script_module() as script_module:

        @hidet.lang.script
        def matmul_kernel(
            a_ptr: ~float32,  # ~ means "pointer to", similar to "*" in C
            b_ptr: ~float32,
            c_ptr: ~float32,
            m_size: int32,
            n_size: int32,
            k_size: int32,
        ):
            attr.func_name = 'matmul_kernel'
            attr.cuda_block_dim = num_threads
            attr.cuda_grid_dim = (
                (m_size + block_m_size - 1) // block_m_size,
                (n_size + block_n_size - 1) // block_n_size,
            )

            a = as_tensor_pointer(a_ptr, float32, [m_size, k_size])
            b = as_tensor_pointer(b_ptr, float32, [k_size, n_size])
            c = as_tensor_pointer(c_ptr, float32, [m_size, n_size])

            smem_a = tensor('shared', float32, shape=[block_m_size, block_k_size])
            smem_b = tensor('shared', float32, shape=[block_k_size, block_n_size])
            regs_c = tensor(
                scope='register',
                dtype=float32,
                # shape will be inferred from the layout automatically,
                # in this case, the shape is [64, 256]
                layout=(
                    local_layout(warps_m, warps_n)
                    * row_layout(warp_m, warp_n)
                    * local_layout(warp_map_m, warp_map_n)
                    * row_layout(thread_m, thread_n)
                ),
            )

            # initialize the registers
            mma_mapping = (
                spatial(warps_m, warps_n)
                .repeat(warp_m, warp_n)
                .spatial(warp_map_m, warp_map_n)
                .repeat(thread_m, thread_n)
            )
            for i, j in mma_mapping.on(threadIdx.x):
                regs_c[i, j] = 0.0

            # iterate over the k tiles
            num_k_tiles = (k_size + block_k_size - 1) // block_k_size
            for k_tile in range(num_k_tiles):

                # load smem_a [block_m_size, block_k_size] from global memory
                for i, k in auto_map(block_m_size, block_k_size, workers=num_threads).on(
                    threadIdx.x
                ):
                    global_i, global_k = (
                        i + blockIdx.x * block_m_size,
                        k + k_tile * block_k_size,
                    )
                    smem_a[i, k] = (
                        a[global_i, global_k]
                        if global_i < m_size and global_k < k_size
                        else 0.0
                    )

                # load smem_b [block_k_size, block_n_size] from global memory
                for k, j in auto_map(block_k_size, block_n_size, workers=num_threads).on(
                    threadIdx.x
                ):
                    global_k, global_j = (
                        k + k_tile * block_k_size,
                        j + blockIdx.y * block_n_size,
                    )
                    smem_b[k, j] = (
                        b[global_k, global_j]
                        if global_k < k_size and global_j < n_size
                        else 0.0
                    )

                # synchronize all threads in the block
                syncthreads()

                # simt matrix multiply accumulate (mma): regs_c = regs_c + smem_a @ smem_b
                for i, j in mma_mapping.on(threadIdx.x):
                    for k in range(block_k_size):
                        regs_c[i, j] += smem_a[i, k] * smem_b[k, j]

                # synchronize all threads in the block
                syncthreads()

            # store regs_c back to global memory
            for i, j in mma_mapping.on(threadIdx.x):
                global_i = i + blockIdx.x * block_m_size
                global_j = j + blockIdx.y * block_n_size
                if global_i < m_size and global_j < n_size:
                    c[global_i, global_j] = regs_c[i, j]

    assert isinstance(matmul_kernel, hidet.ir.Function)  # matmul is a hidet.ir.Function

    ir_module = script_module.ir_module()
    compiled_function: hidet.runtime.CompiledFunction = hidet.driver.build_ir_module(
        ir_module
    )
    return compiled_function


def main():
    func = matmul_simt_kernel()

    for m, n, k in [(1024, 1024, 1024), (333, 444, 555), (1, 12, 13)]:
        a = hidet.randn([m, k], dtype='float32').cuda()
        b = hidet.randn([k, n], dtype='float32').cuda()
        c = hidet.zeros([m, n]).cuda()
        func(a, b, c, m, n, k)
        numpy.testing.assert_allclose(
            actual=c.cpu().numpy(),
            desired=a.cpu().numpy() @ b.cpu().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

        hidet_latency = hidet.utils.benchmark_func(
            lambda: func(a, b, c, m, n, k), repeat=50
        )
        print(f'{m}x{k}x{n}: hidet takes {hidet_latency:.2f} ms')


# %%
#

main()
