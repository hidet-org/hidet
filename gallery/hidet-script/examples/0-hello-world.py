"""
Hello World!
============
"""

# hidet is a deep learning compiler implemented in python
import hidet

# the hidet.lang module implements the hidet script DSL.
from hidet.lang import attrs, printf

# set the cache directory so that you can check the generated code and binary.
hidet.option.cache_dir('./outs/cache')

# a script module is a compilation unit that contains a list of functions defined in it.
with hidet.script_module() as script_module:

    # we use `hidet.script` to decorate a python function to define a hidet script function.
    @hidet.script
    def launch():
        # we use `hidet.lang.attrs` to set the attributes of the function.
        # the following line specify this hidet script function is a public function.
        attrs.func_kind = 'public'

        # print a message to the standard output.
        printf("Hello World!\n")

# build the script module. The returned `module` is an instance of `hidet.runtime.CompiledModule`,
# which contains the compiled binary, and we can call it to execute the compiled binary.
module = script_module.build()

# you can access the source code of the compiled module using `module.source()`.
print(module.source(color=True))

# directly calling the compiled module would invoke the function named `launch` in the script module.
module()
# we can also explicitly specify the function to be invoked using `module['func_name'](args)`.
module['launch']()
