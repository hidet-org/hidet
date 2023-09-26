"""
Scalar Addition
===============
"""
# %%
# In this example, we will show you how to write a program that adds two float32 numbers.

# %%
# We first import ``hidet`` and ``hidet.lang`` module, as well as set the cache directory.
import hidet
from hidet.lang import attrs

hidet.option.cache_dir('./outs/cache')

# %%
# There are a bunch of data types we can use in Hidet Script, and we can access them in ``hidet.lang.types`` module.
# Each scalar data type has both a full name and a short name. For example, the short name of ``float32`` is
# ``f32``. They are equivalent and can be used interchangeably.
from hidet.lang.types import f32


# %%
# In the script function, we defined two parameters ``a`` and ``b`` with data type ``f32``. The return value of the
# function is also ``f32``. In hidet script, it is **required** to annotate the data type of each parameter. If the
# return type is not annotated, it will be treated as ``void`` data type.
with hidet.script_module() as script_module:
    @hidet.script
    # In the following example, the datatype of a and b is 32-bit floating point number (f32),
    # and the function returns a f32 number.
    def launch(a: f32, b: f32) -> f32:
        attrs.func_kind = 'public'

        return a + b

module = script_module.build()

# %%
# We can invoke the compiled module with two float32 numbers as arguments, and it will return the sum of the two
# numbers.
print(module(3.0, 4.0))
