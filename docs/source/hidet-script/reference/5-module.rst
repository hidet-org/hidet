Module
======

Script module
-------------

A script module is a collections of hidet script functions and global variables. It serves as a compilation unit
of hidet. We can use ``hidet.script_module()`` to create a script module. The created script module can be used as
a python context manager like

.. code-block::

    import hidet
    from hidet.lang import attrs
    from hidet.lang.types import f32

    with hidet.script_module() as script_module:
      # define global variables like
      script_module.define_global_var(name='global_var', var_type=f32)
      ...

      # define functions like
      @hidet.script
      def foo():
        attrs.func_kind = 'public'  # the function kind is mandatory
        ...

      # we can define multiple functions in the script module and call each other

    # we can build the script module to get a CompiledModule (hidet.runtime.CompiledModule)
    # that can be invoked in python directly
    module = script_module.build()
