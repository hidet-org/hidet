"""
Data Layout
===========
"""
# %%
import hidet
from hidet.lang import attrs, printf
from hidet.lang.types import tensor, f32

with hidet.script_module() as script_module:
    @hidet.script
    def kernel():
        attrs.func_kind = 'cpu_kernel'

        a = tensor(dtype=f32, shape=[1024, 1024])  # by default, the layout is a row-major layout

        a[0, 0] = 0.0

        printf("a[%d, %d] = %.1f\n", 0, 0, a[0, 0])

module = script_module.build()
module()
