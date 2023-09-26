"""
More Efficient Matrix Multiplication
====================================

In this example, we show you how to write a more efficient matrix multiplication kernel on NVIDIA GPU that uses shared
memory. For simplicity, we omitted some optimizations like software pipelining (see our `paper`_ for more details).

.. _paper: https://dl.acm.org/doi/10.1145/3575693.3575702


Feel free to skip this example if you are not familiar with CUDA programming.

"""
# %%
import torch
import hidet
from hidet.lang import attrs
from hidet.lang import float32, int32
from hidet.lang import as_tensor_pointer, register_tensor, shared_tensor
from hidet.lang.cuda import threadIdx, blockIdx, syncthreads
from hidet.lang.mapping import spatial, auto_map
from hidet.lang.layout import row_major, local_layout

# the hyperparameters of the kernel
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

    @hidet.script
    def relu(x: float32) -> float32:
        return x if x > 0.0 else 0.0

    @hidet.script
    def matmul_relu_kernel(
        a_ptr: ~float32, b_ptr: ~float32, c_ptr: ~float32,
        m_size: int32, n_size: int32, k_size: int32,
    ):
        attrs.func_name = 'matmul_kernel'
        attrs.cuda.block_dim = num_threads
        attrs.cuda.grid_dim = (
            (m_size + block_m_size - 1) // block_m_size,
            (n_size + block_n_size - 1) // block_n_size,
        )

        a = as_tensor_pointer(a_ptr, float32, [m_size, k_size])
        b = as_tensor_pointer(b_ptr, float32, [k_size, n_size])
        c = as_tensor_pointer(c_ptr, float32, [m_size, n_size])

        # define tensors in shared memory
        smem_a = shared_tensor(float32, shape=[block_m_size, block_k_size])
        smem_b = shared_tensor(float32, shape=[block_k_size, block_n_size])

        # define the accumulation tensor in register
        regs_c = register_tensor(
            dtype=float32,
            # shape will be inferred from the layout automatically,
            # in this case, the shape is [64, 256]
            layout=(
                local_layout(warps_m, warps_n)
                * row_major(warp_m, warp_n)
                * local_layout(warp_map_m, warp_map_n)
                * row_major(thread_m, thread_n)
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
                c[global_i, global_j] = relu(regs_c[i, j])


module = script_module.build()


def hidet_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    m_size, n_size, k_size = a.shape[0], b.shape[1], a.shape[1]
    c = torch.empty([m_size, n_size], device='cuda')
    module(a, b, c, m_size, n_size, k_size)
    return c


def torch_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    return torch.matmul(a, b).relu()


# %%
# Run the program with different input sizes. This implementation archives about 30% performance of cuBLAS kernels.
# For more efficient implementations, please refer to the `ones`_ in hidet package.
#
# .. _ones: https://github.com/hidet-org/hidet/tree/main/python/hidet/graph/ops/matmul

for m, n, k in [(1024, 1024, 1024), (256, 256, 256), (32, 32, 32)]:
    a = torch.randn(m, k, dtype=torch.float32, device='cuda')
    b = torch.randn(k, n, dtype=torch.float32, device='cuda')

    c1 = hidet_matmul_relu(a, b)
    c2 = torch_matmul_relu(a, b)

    torch.testing.assert_close(c1, c2, atol=1e-4, rtol=1e-4)

    hidet_latency = hidet.utils.benchmark_func(
        lambda: hidet_matmul_relu(a, b), repeat=50
    )
    print(f'{m}x{k}x{n}:')
    print(' torch: {:.3f} ms'.format(hidet.utils.benchmark_func(lambda: torch_matmul_relu(a, b))))
    print(' hidet: {:.3f} ms'.format(hidet.utils.benchmark_func(lambda: hidet_matmul_relu(a, b))))

# %%
# Get the source code:
print(module.source())

