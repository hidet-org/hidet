import pytest

from hidet.backend import build
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import tensor_var
from hidet.ir.func import IRModule
from hidet.ir.stmt import BlackBoxStmt, BufferStoreStmt
from hidet.ir.task import Grid
from hidet.utils import cuda


def test_lds128():
    with FunctionBuilder('test_lds128.grid', attrs={'worker': Grid(grid_dim=1, block_dim=1)}) as fb:
        # params
        regs_tensor = tensor_var('regs_tensor', [4], 'register', 'float32')
        smem_tensor = tensor_var('smem_tensor', [4], 'shared', 'float32')
        fb.extend_local_vars([regs_tensor, smem_tensor])

        # body
        sb = StmtBuilder()
        for i in range(4):
            sb += BufferStoreStmt(smem_tensor, [i], i)
        for i in range(4):
            sb += BufferStoreStmt(regs_tensor, [i], smem_tensor[i])
        sb += BlackBoxStmt(r'printf("%.2f %.2f %.2f %.2f\n", {}, {}, {}, {});',
                           regs_tensor[0], regs_tensor[1], regs_tensor[2], regs_tensor[3])
        fb.set_body(sb.finish())

    func = fb.get()
    ir_module = IRModule({func.name: func}, task=None)
    module = build(ir_module, './outs/test_lds128')
    module['test_lds128']()
    cuda.device_synchronize()


def test_sts128():
    with FunctionBuilder('test_sts128.grid', attrs={'worker': Grid(grid_dim=1, block_dim=1)}) as fb:
        # params
        regs_tensor = tensor_var('regs_tensor', [4], 'register', 'float32')
        smem_tensor = tensor_var('smem_tensor', [4], 'shared', 'float32')
        fb.extend_local_vars([regs_tensor, smem_tensor])

        # body
        sb = StmtBuilder()
        for i in range(4):
            sb += BufferStoreStmt(regs_tensor, [i], i)
        for i in range(4):
            sb += BufferStoreStmt(smem_tensor, [i], regs_tensor[i])
        sb += BlackBoxStmt(r'printf("%.2f %.2f %.2f %.2f\n", {}, {}, {}, {});',
                           smem_tensor[0], smem_tensor[1], smem_tensor[2], smem_tensor[3])
        fb.set_body(sb.finish())

    func = fb.get()
    ir_module = IRModule({func.name: func}, task=None)
    module = build(ir_module, './outs/test_sts128')
    module['test_sts128']()
    cuda.device_synchronize()


if __name__ == '__main__':
    pytest.main(__file__)
