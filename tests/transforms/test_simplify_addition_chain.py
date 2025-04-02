import hidet
from hidet.lang import attrs
from hidet.ir.dtypes import int32
from hidet.transforms import simplify_addition_chain_pass


def test_div_mod_cancellation():
    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: int32, b: int32):
            attrs.func_kind = 'cuda_kernel'
            c = a / 10 * 10 + b + a % 10

    module = script_module.ir_module()
    print(module)
    """
    def func(
        a: int32,
        b: int32
    )
        # kind: cuda_kernel
        # func_kind: cuda_kernel
        declare c: int32 = ((((a / 10) * 10) + b) + (a % 10))
    """
    transform = simplify_addition_chain_pass()
    transformed_module = transform(module)
    print(transformed_module)
    """
    def func(
        a: int32,
        b: int32
    )
        # kind: cuda_kernel
        # func_kind: cuda_kernel
        declare c: int32 = (a + b)
    """

    assert "declare c: int32 = (a + b)" in str(transformed_module)


def test_reorder():
    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: int32, b: int32):
            attrs.func_kind = 'cuda_kernel'
            for c in range(10):
                d = a + c + b

    module = script_module.ir_module()
    print(module)
    """
    def func(
        a: int32,
        b: int32
    )
        # kind: cuda_kernel
        # func_kind: cuda_kernel
        for c in range(10):
            declare d: int32 = ((a + c) + b)
    """
    transform = simplify_addition_chain_pass()
    transformed_module = transform(module)
    print(transformed_module)
    """
    def func(
        a: int32,
        b: int32
    )
        # kind: cuda_kernel
        # func_kind: cuda_kernel
        for c in range(10):
            declare d: int32 = ((a + b) + c)
    """
    assert "(a + b) + c" in str(transformed_module) or "a + b + c" in str(transformed_module)


if __name__ == '__main__':
    test_div_mod_cancellation()
    test_reorder()
