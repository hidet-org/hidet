"""
Vector Addition
===============
"""
import hidet
from hidet.lang import attrs
from hidet.lang.types import f32

hidet.option.cache_dir('./outs/cache')

with hidet.script_module() as script_module:
    @hidet.script
    # we can use dtype[shape] like "f32[3]", "f32[3, 4]", or "f32[(3, 4)]" to define a tensor
    # with given shape and data type. If no return type is annotated, the return type will be
    # void (hidet.lang.types.void).
    def launch(a: f32[3], b: f32[3], c: f32[3]):
        attrs.func_kind = 'public'

        # we can use the pythonic for-range syntax to iterate over a range
        for i in range(10):
            c[i] = a[i] + b[i]


module = script_module.build()

# create the input and output tensors (on cpu, with f32 data type by default)
a = hidet.randn([3])
b = hidet.randn([3])
c = hidet.empty([3])

# we can directly call the compiled module with the input and output tensors
module(a, b, c)
print(a)
print(b)
print(c)
