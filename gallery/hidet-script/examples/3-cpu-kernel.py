"""
CPU Kernel
==========
"""
import hidet
from hidet.lang import attrs
from hidet.lang.types import f32

hidet.option.cache_dir('./outs/cache')

with hidet.script_module() as script_module:
    @hidet.script
    def matmul(a: f32[16, 16], b: f32[16, 16], c: f32[16, 16]):
        # 'cpu_kernel' is a kernel function on cpu. In the compiled module, only 'public'
        # functions will be exposed to the outside. By default, if there is a kernel function
        # (i.e., a function with `attrs.func_kind` set to 'cpu_kernel' or 'cuda_kernel'),
        # a default launch function with 'public' func_kind will be generated to launch
        # the kernel function.
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
