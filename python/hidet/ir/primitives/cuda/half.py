from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import attrs, script, asm, cast  # pylint: disable=import-outside-toplevel
    from hidet.lang.types import uint32, void_p

    @script
    def sub_f16x2_(d: void_p, a: uint32, b: uint32):
        attrs.func_kind = 'cuda_internal'
        attrs.func_name = 'sub_f16x2'

        asm('sub.f16x2 %0, %1, %2;', outputs=[cast(d, ~uint32)[0]], inputs=[a, b], is_volatile=True)

    @script
    def fma_f16x2_(d: void_p, a: uint32, b: uint32, c: uint32):
        attrs.func_kind = 'cuda_internal'
        attrs.func_name = 'fma_f16x2'

        asm('fma.rn.f16x2 %0, %1, %2, %3;', outputs=[cast(d, ~uint32)[0]], inputs=[a, b, c], is_volatile=True)

    funcs = [sub_f16x2_, fma_f16x2_]
    for func in funcs:
        assert isinstance(func, Function)
        register_primitive_function(name=func.name, func_or_type=func)


def sub_f16x2(d: Expr, a: Expr, b: Expr):
    """
    Subtract two f16x2 values and store the result in `d`.

    Expect `d` to be an uint32 pointer while `a` an `b` are uint32 values, all of them will be interpreted as f16x2.

    Parameters
    ----------
    d: Expr
        The pointer to the f16x2 result, stored with uint32 data type.
    a: Expr
        The first f16x2 operand stored with uint32 data type.
    b: Expr
        The second f16x2 operand stored with uint32 data type.
    """
    return call_primitive_func('sub_f16x2', args=[d, a, b])


def fma_f16x2(d: Expr, a: Expr, b: Expr, c: Expr):
    """
    Multiply two f16x2 values and add the third f16x2 value and store the result in `d`.

    Expect `d` to be an uint32 pointer while `a`, `b`, and `c` are uint32 values, all of them will be interpreted as
    f16x2.

    Parameters
    ----------
    d: Expr
        The pointer to the f16x2 result, stored with uint32 data type.
    a: Expr
        The first f16x2 operand stored with uint32 data type.
    b: Expr
        The second f16x2 operand stored with uint32 data type.
    c: Expr
        The third f16x2 operand stored with uint32 data type.
    """
    return call_primitive_func('fma_f16x2', args=[d, a, b, c])
