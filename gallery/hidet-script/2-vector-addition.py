"""
Vector Addition
===============
"""
# %%
# In this example, we will show you how to write a program that adds two float32 vectors in hidet script.
import hidet
from hidet.lang import attrs
from hidet.lang.types import f32

hidet.option.cache_dir('./outs/cache')

# %%
# In the script function, we annotate the data type of parameter ``a``, ``b``, and ``c`` as ``f32[3]``, which means
# a 3-element vector of 32-bit floating point numbers. In general, we can use ``dtype[shape]`` to define a tensor with
# given shape and data type. For example, ``f32[3, 4]`` is a 3x4 float32 matrix, and ``int32[3, 4, 5]`` is a 3x4x5 int32
# tensor.
#
# We can use ``for i in range(extent)`` to iterate over a range, where ``extent`` is the extent of the loop.
with hidet.script_module() as script_module:
    @hidet.script
    def launch(a: f32[3], b: f32[3], c: f32[3]):
        attrs.func_kind = 'public'

        for i in range(10):
            c[i] = a[i] + b[i]

module = script_module.build()

# %%
# Create the input and output tensors (on cpu, with f32 data type by default):
a = hidet.randn([3])
b = hidet.randn([3])
c = hidet.empty([3])

# %%
# Call the compiled module with the input and output tensors
module(a, b, c)
print(a)
print(b)
print(c)
