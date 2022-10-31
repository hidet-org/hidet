import pytest

from hidet.backend import build
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.type import FuncType, VoidType
from hidet.ir.expr import var, tensor_var
from hidet.ir.func import IRModule
from hidet.ir.primitives import lds128, sts128
from hidet.ir.stmt import BlackBoxStmt, AssignStmt, BufferStoreStmt, DeclareStmt, DeclareScope
from hidet.utils import cuda
from hidet.driver import build_ir_module
from hidet.transforms.tools import fuse_and_pack


def test_lds128(capfd):
    with FunctionBuilder('test_lds128_grid', kind='cuda_kernel', grid_dim=1, block_dim=1) as fb:
        # params
        regs = [var(f'reg{i}', 'float32') for i in range(4)]
        smem_tensor = tensor_var('smem_tensor', [4], 'float32')
        fb += DeclareStmt(smem_tensor, scope=DeclareScope.Shared)
        for reg in regs:
            fb += DeclareStmt(reg)

        # body
        for i in range(4):
            fb += BufferStoreStmt(smem_tensor, [i], i)
        fb += lds128(regs[0], regs[1], regs[2], regs[3], smem_tensor)
        fb += BlackBoxStmt(r'printf("%.2f %.2f %.2f %.2f\n", {}, {}, {}, {});', regs[0], regs[1], regs[2], regs[3])
        fb.set_body(fb.finish())

    func = fb.get()
    ir_module = IRModule({func.name: func}, task=None)
    fuse_and_pack(ir_module, func, pack_func_name='test_lds128')
    compiled_func = build_ir_module(
        ir_module, func_name='test_lds128', working_dir='./outs/', func_type=FuncType([], VoidType())
    )
    compiled_func()
    cuda.device_synchronize()
    captured = capfd.readouterr()
    assert captured.out == '0.00 1.00 2.00 3.00\n'


def test_sts128(capfd):
    with FunctionBuilder('test_sts128_grid', kind='cuda_kernel', grid_dim=1, block_dim=1) as fb:
        # params
        regs = [var(f'reg{i}', 'float32') for i in range(4)]
        smem_tensor = tensor_var('smem_tensor', [4], 'float32')
        fb += DeclareStmt(smem_tensor, scope=DeclareScope.Shared)
        for reg in regs:
            fb += DeclareStmt(reg)

        # body
        for i in range(4):
            fb += AssignStmt(regs[i], i)
        fb += sts128(regs[0], regs[1], regs[2], regs[3], smem_tensor)
        fb += BlackBoxStmt(
            r'printf("%.2f %.2f %.2f %.2f\n", {}, {}, {}, {});',
            smem_tensor[0],
            smem_tensor[1],
            smem_tensor[2],
            smem_tensor[3],
        )

    func = fb.get()
    ir_module = IRModule({func.name: func}, task=None)
    fuse_and_pack(ir_module, func, pack_func_name='test_sts128')
    compiled_func = build_ir_module(
        ir_module, func_name='test_sts128', working_dir='./outs/', func_type=FuncType([], VoidType())
    )
    compiled_func()
    cuda.device_synchronize()
    captured = capfd.readouterr()
    assert captured.out == '0.00 1.00 2.00 3.00\n'


if __name__ == '__main__':
    pytest.main(__file__)
