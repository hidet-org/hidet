"""
Naive Matrix Multiplication
===========================
"""
import torch
import hidet
from hidet.lang import attrs, printf
from hidet.lang.types import f32

# the `hidet.lang.cuda` module defines the primitives provided by cuda platform
from hidet.lang.cuda import threadIdx, blockIdx

hidet.option.cache_dir('./outs/cache')

m_size, n_size, k_size = 1024, 1024, 1024

with hidet.script_module() as script_module:
    @hidet.script
    # hidet script is very flexible. You could use the python variables defined outside the
    # hidet script function.
    def matmul(a: f32[m_size, k_size], b: f32[k_size, n_size], c: f32[m_size, n_size]):
        # 'cuda_kernel' is another function type that indicates this function is a cuda kernel
        # (which corresponds to the __global__ function in CUDA C, if you are familiar with it).
        attrs.func_kind = 'cuda_kernel'

        # we can specify the attributes of the cuda kernel with `attrs.cuda.*` attributes
        # the following two lines specify the grid dimension and thread block dimension.
        attrs.cuda.block_dim = [16, 16]
        attrs.cuda.grid_dim = [(m_size + 15) // 16, (n_size + 15) // 16]

        i = blockIdx.x * 16 + threadIdx.x
        j = blockIdx.y * 16 + threadIdx.y

        if i < m_size and j < n_size:
            c[i, j] = 0.0
            for k in range(k_size):
                c[i, j] += a[i, k] * b[k, j]


module = script_module.build()

# define three cuda tensors
a = hidet.randn([m_size, k_size], device='cuda')
b = hidet.randn([k_size, n_size], device='cuda')
c = hidet.empty([m_size, n_size], device='cuda')

module(a, b, c)

# check the correctness of the result
torch.testing.assert_close(c.torch(), a.torch() @ b.torch(), atol=1e-4, rtol=1e-4)
