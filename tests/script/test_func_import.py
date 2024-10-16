import pytest
import hidet


def test_function_import():
    from hidet.lang import attrs, printf

    # function decorated by `hidet.script` outside any `script_module` context will not
    # be added to any script module. The function can be defined in another python module
    @hidet.script
    def outside_func():
        attrs.func_kind = 'public'

        printf("outside\n")

    with hidet.script_module() as script_module:

        # function decorated by `hidet.script` within a `script_module` context will be
        # added to the script module.
        @hidet.script
        def inside_func():
            attrs.func_kind = 'public'

            printf("inside\n")

        @hidet.script
        def launch():
            attrs.func_kind = 'public'

            inside_func()
            outside_func()  # directly call function defined outside the script module

    print(script_module.ir_module())
    """
    def inside_func()
        # kind: public
        printf("inside\n");

    def outside_func()
        # kind: public
        printf("outside\n");

    def launch()
        # kind: public
        inside_func()
        outside_func()
    """

    built = script_module.build()
    built()
    """
    inside
    outside
    """
