import pytest

from hidet.backend import build
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import var
from hidet.ir.func import IRModule
from hidet.ir.primitives import thread_idx
from hidet.ir.stmt import AsmStmt, BlackBoxStmt
from hidet.ir.task import Grid


def test_asm_stmt():
    with FunctionBuilder('demo_asm.grid', attrs={'worker': Grid(grid_dim=1, block_dim=1)}) as fb:
        # body
        sb = StmtBuilder()
        with sb.let('a', thread_idx()) as a:
            with sb.let('b', thread_idx()) as b:
                c = var('c')
                fb.extend_local_vars([c])
                sb += AsmStmt(
                    r'add.s32 %0, %1, %2;',
                    outputs=[('=r', c)],
                    inputs=[('r', a), ('r', b)],
                    is_volatile=False
                )
                sb += BlackBoxStmt(r'printf("%d %d %d\n", {}, {}, {});', a, b, c)
        fb.set_body(sb.finish())
    func = fb.get()
    ir_module = IRModule({func.name: func}, task=None)
    module = build(ir_module, './outs/demo_asm')
    module['demo_asm']()


if __name__ == '__main__':
    pytest.main(__file__)
