from hidet.ir.expr import Expr, cast
from hidet.ir.stmt import asm
from hidet.ir.dtypes import uint32


def lop3(d: Expr, a: Expr, b: Expr, c: Expr, *, imm_lut: int):
    """
    Perform a logical operation on three 32-bit values and store the result in `d`.

    The logical operation is determined by the immediate value `imm_lut`.

    See the PTX ISA documentation for the `lop3` instruction for more information:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3

    Parameters
    ----------
    d: Expr
        The pointer to the 32-bit result.
    a: Expr
        The first 32-bit operand.
    b: Expr
        The second 32-bit operand.
    c: Expr
        The third 32-bit operand.
    imm_lut: int
        The immediate value that determines the logical operation. Given logical operation `f(a, b, c)`, the
        immediate value `imm_lut` should be set to `f(0xF0, 0xCC, 0xAA)` to indicate the logical operation.
    """
    assert 0 <= imm_lut <= 255

    return asm(
        'lop3.b32 %0, %1, %2, %3, {};'.format(imm_lut),
        outputs=[cast(d, ~uint32)[0]],
        inputs=[a, b, c, imm_lut],
        is_volatile=True,
    )
