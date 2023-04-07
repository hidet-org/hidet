# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

import hidet
from hidet.backend import build
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.type import FuncType, VoidType
from hidet.ir.expr import var, tensor_var
from hidet.ir.func import IRModule
from hidet.ir.primitives import lds128, sts128
from hidet.ir.stmt import BlackBoxStmt, AssignStmt, BufferStoreStmt, DeclareStmt, DeclareScope
from hidet.driver import build_ir_module


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
    compiled_func = build_ir_module(ir_module, output_dir='./outs/')
    compiled_func()
    hidet.cuda.synchronize()
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
    compiled_func = build_ir_module(ir_module, output_dir='./outs/')
    compiled_func()
    hidet.cuda.synchronize()
    captured = capfd.readouterr()
    assert captured.out == '0.00 1.00 2.00 3.00\n'


if __name__ == '__main__':
    pytest.main(__file__)
