"""
Kernel Functions
================
"""
# %%
# Besides the ``public`` function, there are other function kinds in hidet script. Currently, we support the following
# function kinds:
#
# - ``public``: a public function. The public functions in a script module will be exposed to the outside and can be
#   invoked by the outside (in our case, we can call them in python).
# - ``cpu_kernel``: a kernel function on cpu.
# - ``cpu_internal``: an internal function on cpu.
# - ``cuda_kernel``: a kernel function on cuda.
# - ``cuda_internal``: an internal function on cuda.
#
# .. tip::
#   :class: margin
#
#   The ``cuda_kernel`` and ``cuda_internal`` correspond to the ``__global__`` and ``__device__`` functions in CUDA.
#
# Usually, we use the ``cpu_kernel`` and ``cuda_kernel`` to define the kernel functions. The ``cpu_internal`` and
# ``cuda_internal`` are used to define the internal functions that are only used by the kernel functions.
#
# When there is only one kernel function in a script module and there is no function named ``launch``, a default
# ``launch`` function will be generated to launch the kernel function.
#

# %%
# CPU kernel function
# -------------------
import hidet
from hidet.lang import attrs
from hidet.lang.types import f32

hidet.option.cache_dir('./outs/cache')

with hidet.script_module() as script_module:
    @hidet.script
    def matmul(a: f32[16, 16], b: f32[16, 16], c: f32[16, 16]):
        # specify the function kind as 'cpu_kernel'
        attrs.func_kind = 'cpu_kernel'

        for i in range(16):
            for j in range(16):
                c[i, j] = 0.0
                for k in range(16):
                    c[i, j] += a[i, k] * b[k, j]


module = script_module.build()

a = hidet.randn([16, 16])
b = hidet.randn([16, 16])
c = hidet.empty([16, 16])

module(a, b, c)

# %%
# We can check the generated source code to see that the ``launch`` function is generated automatically.
print(module.source())


# %%
# CUDA kernel function
# --------------------
# We can also define a kernel function on CUDA. The following example defines a kernel function on cuda.
#
# We can access cuda primitive variables and functions in the ``hidet.lang.cuda`` module.
from hidet.lang.cuda import blockIdx, threadIdx, blockDim

# workload size
m_size = 1024
n_size = 1024
k_size = 1024

with hidet.script_module() as script_module:
    @hidet.script
    def matmul(a: f32[m_size, k_size], b: f32[k_size, n_size], c: f32[m_size, n_size]):
        # specify the function kind as 'cuda_kernel'
        attrs.func_kind = 'cuda_kernel'

        # specify the grid dimension and block dimension
        attrs.cuda.grid_dim = (m_size + 15) // 16, (n_size + 15) // 16
        attrs.cuda.block_dim = 16, 16

        # the coordinate of the c matrix that this thread is responsible for
        i = blockIdx.x * blockDim.x + threadIdx.x
        j = blockIdx.y * blockDim.y + threadIdx.y

        if i < m_size and j < n_size:
            c[i, j] = 0.0
            for k in range(k_size):
                c[i, j] += a[i, k] * b[k, j]


module = script_module.build()

a = hidet.randn([m_size, k_size], device='cuda')
b = hidet.randn([k_size, n_size], device='cuda')
c = hidet.empty([m_size, n_size], device='cuda')

module(a, b, c)

# compare the result with torch.matmul
hidet.utils.assert_close(c, a.torch() @ b.torch(), atol=1e-4, rtol=1e-4)

# %%
# We can check the generated source code:
#
# .. tip::
#    :class: margin
#
#    You can find that there is no boundary checking in the kernel function. This is because hidet infers the value
#    range for each index variable and finds that the if condition is always true, so it simplifies the if-statement.
print(module.source())

