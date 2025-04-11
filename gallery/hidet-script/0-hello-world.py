"""
Hello World!
============

In this example, we will show you how to use hidet to write a simple "Hello World" program.

"""

# %%
# Hidet is a deep learning compiler implemented in python. Let's import it first.
import hidet

# %%
# Hidet caches all its generated source code and binary in its cache directory. We can set the cache directory
# to a local directory ``./outs/cache`` so that you can check the generated code and binary.
hidet.option.cache_dir('./outs/cache')

# %%
# The ``hidet.lang`` submodule implements the Hidet Script domain specific language.
# In this example, we will use ``attrs`` variable and ``printf`` function from ``hidet.lang``.
from hidet.lang import attrs, printf

# %%
# A **script module** is a compilation unit that contains a list of functions defined in it. Inside a script module,
# we can use ``hidet.script`` to define a hidet script function. The following example defines a function named
# ``launch`` that prints a message to the standard output.

with hidet.script_module() as script_module:

    # we use `hidet.script` to decorate a python function to define a hidet script function.
    @hidet.script
    def launch():
        # we use `hidet.lang.attrs` to set the attributes of the function.
        # the following line specify this hidet script function is a public function.
        attrs.func_kind = 'public'

        # print a message to the standard output.
        printf("Hello World!\n")


# %%
# With the script module defined, we can build the script module with ``build()`` method. The returned ``module`` is
# an instance of ``hidet.runtime.CompiledModule``, which contains the compiled binary.
module = script_module.build()

# %%
# We can directly call the compiled module, in this case the 'launch' function would be invoked.
#
# .. note::
#    :class: margin
#
#    The printed message has not been captured by our documentation generation tool (i.e., sphinx).
#    If you run the script by yourself, you will see the message printed out in your console.
module()

# %%
# We can also explicitly specify the function to be invoked using ``module['func_name'](args)``.
module['launch']()

# %%
# you can access the source code of the compiled module using ``module.source()``.
#
# .. note::
#    :class: margin
#
#    The function in the source code has a prefix ``hidet_``, which is used to avoid name conflict with standard
#    library functions.
print(module.source(color=True))
