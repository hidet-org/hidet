"""
Naive Matrix Multiplication
===========================
"""

# %%
# In this example, we will show you how to write a program that performs matrix multiplication on GPU that supports
# arbitrary input size.
import torch
import hidet
from hidet.lang import attrs
from hidet.lang.types import f32, i32, tensor_pointer
from hidet.lang.cuda import threadIdx, blockIdx

hidet.option.cache_dir('./outs/cache')

with hidet.script_module() as script_module:

    @hidet.script
    def matmul_kernel(a_ptr: ~f32, b_ptr: ~f32, c_ptr: ~f32, m_size: i32, n_size: i32, k_size: i32):
        attrs.func_kind = 'cuda_kernel'
        attrs.cuda.block_dim = 16, 16
        attrs.cuda.grid_dim = (m_size + 15) // 16, (n_size + 15) // 16

        # define three tensor pointers that hold the shape and dtype information
        a = tensor_pointer(dtype=f32, shape=[m_size, k_size], init=a_ptr)
        b = tensor_pointer(dtype=f32, shape=[k_size, n_size], init=b_ptr)
        c = tensor_pointer(dtype=f32, shape=[m_size, n_size], init=c_ptr)

        i = blockIdx.x * 16 + threadIdx.x
        j = blockIdx.y * 16 + threadIdx.y

        if i < m_size and j < n_size:
            c[i, j] = 0.0
            for k in range(k_size):
                c[i, j] += a[i, k] * b[k, j]


module = script_module.build()


# %%
# Hidet compiled module can be called directly with pytorch tensors.


def matmul(a: torch.Tensor, b: torch.Tensor):
    m_size, n_size, k_size = a.shape[0], b.shape[1], a.shape[1]
    c = torch.empty([m_size, n_size], device='cuda')
    module(a, b, c, m_size, n_size, k_size)
    return c


# %%
# Run the compiled kernels with different input sizes and check the correctness of the result.
for m_size, n_size, k_size in [(234, 345, 567), (123, 456, 789)]:
    a = torch.randn(m_size, k_size, device='cuda')
    b = torch.randn(k_size, n_size, device='cuda')

    c1 = matmul(a, b)
    c2 = torch.matmul(a, b)

    # check the correctness of the result
    torch.testing.assert_close(c1, c2, atol=1e-4, rtol=1e-4)


# %%
print(module.source())
