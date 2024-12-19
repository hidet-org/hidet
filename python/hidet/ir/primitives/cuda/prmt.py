from typing import Optional

from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


def resolve_func_name(mode: Optional[str] = None) -> str:
    if mode is None:
        return 'prmt_b32'
    else:
        return 'prmt_b32_{}'.format(mode)


def resolve_inst_template(mode: Optional[str] = None) -> str:
    if mode is None:
        return 'prmt.b32 %0, %1, %2, %3;'
    else:
        return 'prmt.b32.{} %0, %1, %2, %3;'.format(mode)


@initialize()
def register_functions():
    from hidet.lang import attrs, script, asm, cast  # pylint: disable=import-outside-toplevel
    from hidet.lang.types import uint32, void_p

    for mode in [None, 'f4e', 'b4e', 'rc8', 'ecl', 'ecr', 'rc16']:
        template = resolve_inst_template(mode)

        @script
        def prmt_primitive(d: void_p, a: uint32, b: uint32, c: uint32):
            attrs.func_kind = 'cuda_internal'
            attrs.func_name = resolve_func_name(mode)

            asm(template, outputs=[cast(d, ~uint32)[0]], inputs=[a, b, c], is_volatile=True)

        assert isinstance(prmt_primitive, Function)
        register_primitive_function(name=prmt_primitive.name, func_or_type=prmt_primitive)


def prmt(d: Expr, a: Expr, b: Expr, c: Expr, *, mode: Optional[str] = None):
    """
    Perform a byte-level permutation operation on two 32-bit values and store the result in `d`.

    The permutation operation is determined by the permutation mode `mode`.

    See Also the PTX ISA documentation for the `prmt` instruction for more information:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt

    Parameters
    ----------
    d: Expr
        The pointer to the 32-bit result.
    a: Expr
        The first uint32 operand.
    b: Expr
        The second uint32 operand.
    c: Expr
        The control operand.
    mode: Optional[str]
        The permutation mode. If not provided, the default mode is used.
    """
    assert mode in [None, 'f4e', 'b4e', 'rc8', 'ecl', 'ecr', 'rc16']
    return call_primitive_func(resolve_func_name(mode), args=[d, a, b, c])
