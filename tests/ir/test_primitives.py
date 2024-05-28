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
from hidet.ir.module import IRModule
from hidet.ir.primitives import lds128, sts128, lds64, lds32, sts64, sts32
from hidet.ir.stmt import BlackBoxStmt, AssignStmt, BufferStoreStmt, DeclareStmt, DeclareScope
from hidet.drivers import build_ir_module


@pytest.mark.parametrize("load_bits", [128, 64, 32])
def test_lds(load_bits, capfd):
    from hidet.ir.dtypes import u32
    from hidet.ir.type import ReferenceType

    nr_regs = load_bits // (u32.nbytes * 8)
    with FunctionBuilder(f"test_lds{load_bits}_grid", kind="cuda_kernel", grid_dim=1, block_dim=1) as fb:
        # params
        regs = [var(f"reg{i}", "float32") for i in range(nr_regs)]
        smem_tensor = tensor_var("smem_tensor", [nr_regs], "float32")
        fb += DeclareStmt(smem_tensor, scope=DeclareScope.Shared)
        for reg in regs:
            fb += DeclareStmt(reg)

        # body
        for i in range(nr_regs):
            fb += BufferStoreStmt(smem_tensor, [i], i)
        reg_vars = [~i for i in regs]
        if load_bits == 128:
            lds = lds128
        elif load_bits == 64:
            lds = lds64
        elif load_bits == 32:
            lds = lds32
        fb += lds(*reg_vars, smem_tensor)
        fmt = " ".join(["%.2f" for i in range(nr_regs)])
        var_args = ", ".join([r"{}" for i in range(nr_regs)])
        fb += BlackBoxStmt(f'printf("{fmt}\\n", {var_args});', *regs)
        fb.set_body(fb.finish())

    func = fb.get()
    ir_module = IRModule(functions={func.name: func})
    compiled_func = ir_module.build()
    compiled_func()
    hidet.cuda.synchronize()
    captured = capfd.readouterr()
    expected = " ".join(["{:.2f}".format(float(i)) for i in range(nr_regs)])
    assert captured.out == expected + "\n"


@pytest.mark.parametrize("store_bits", [128, 64, 32])
def test_sts(store_bits, capfd):
    from hidet.ir.dtypes import u32
    from hidet.ir.type import ReferenceType

    nr_regs = store_bits // (u32.nbytes * 8)
    with FunctionBuilder(f"test_sts{store_bits}_grid", kind="cuda_kernel", grid_dim=1, block_dim=1) as fb:
        # params
        regs = [var(f"reg{i}", "float32") for i in range(nr_regs)]
        smem_tensor = tensor_var("smem_tensor", [nr_regs], "float32")
        fb += DeclareStmt(smem_tensor, scope=DeclareScope.Shared)
        for reg in regs:
            fb += DeclareStmt(reg)

        # body
        for i in range(nr_regs):
            fb += AssignStmt(regs[i], i)

        reg_vars = [~i for i in regs]
        if store_bits == 128:
            sts = sts128
        elif store_bits == 64:
            sts = sts64
        elif store_bits == 32:
            sts = sts32
        fb += sts(*reg_vars, smem_tensor)
        fmt = " ".join(["%.2f" for i in range(nr_regs)])
        var_args = ", ".join([r"{}" for i in range(nr_regs)])
        smem_vars = [smem_tensor[i] for i in range(nr_regs)]
        fb += BlackBoxStmt(f'printf("{fmt}\\n", {var_args});', *smem_vars)
        fb.set_body(fb.finish())

    func = fb.get()
    ir_module = IRModule(functions={func.name: func})
    compiled_func = ir_module.build()
    compiled_func()
    hidet.cuda.synchronize()
    captured = capfd.readouterr()
    expected = " ".join(["{:.2f}".format(float(i)) for i in range(nr_regs)])
    assert captured.out == expected + "\n"


if __name__ == "__main__":
    pytest.main(__file__)
