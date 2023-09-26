"""
Scalar Addition
===============
"""
import hidet
from hidet.lang import attrs

# there are a bunch of data types we can use in hidet.lang.types module
from hidet.lang.types import f32

hidet.option.cache_dir('./outs/cache')

with hidet.script_module() as script_module:
    @hidet.script
    # the parameters and return value of each hidet script function must be annotated with
    # their data types. In the following example, the data-type of a and b is 32-bit floating
    # point number (f32), and the function returns a f32 number.
    def launch(a: f32, b: f32) -> f32:
        attrs.func_kind = 'public'

        return a + b

module = script_module.build()

# returns 7.0
print(module(3.0, 4.0))
