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
import numpy as np
import pytest

import hidet
from hidet.driver import build_ir_module
from hidet.ir.builders import FunctionBuilder
from hidet.ir.expr import Var, tensor_var
from hidet.ir.func import IRModule
from hidet.ir.primitives.cuda import thread_idx
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_sync, mma_configs
from hidet.ir.stmt import BufferStoreStmt, DeclareStmt, DeclareScope
from hidet.ir.type import tensor_pointer_type, FuncType, data_type


def matmul_mma_tensor_core(config: MmaConfig):
    with FunctionBuilder(name='matmul_mma_grid', kind='cuda_kernel', grid_dim=1, block_dim=32) as fb:
        # parameters
        a = Var('a', tensor_pointer_type(config.input_dtype, [1, config.m, config.k]))
        b = Var('b', tensor_pointer_type(config.input_dtype, [1, config.k, config.n]))
        c = Var('c', tensor_pointer_type(config.output_dtype, [1, config.m, config.n]))
        fb.extend_params([a, b, c])

        # local variables
        regs_a = tensor_var('regs_a', [config.a_elements], config.input_dtype)
        regs_b = tensor_var('regs_b', [config.b_elements], config.input_dtype)
        regs_c = tensor_var('regs_c', [config.c_elements], config.output_dtype)
        fb += DeclareStmt(regs_a)
        fb += DeclareStmt(regs_b)
        fb += DeclareStmt(regs_c)

        # body
        w = thread_idx()
        for p in range(config.c_elements):
            fb += BufferStoreStmt(regs_c, [p], 0.0)
        for p, (i, k) in enumerate(config.a_load_map(w)):
            fb += BufferStoreStmt(regs_a, [p], a[0, i, k])
        for p, (k, j) in enumerate(config.b_load_map(w)):
            fb += BufferStoreStmt(regs_b, [p], b[0, k, j])
        fb += mma_sync(config, regs_a, regs_b, regs_c)
        for p, (i, j) in enumerate(config.c_store_map(w)):
            fb += BufferStoreStmt(c, [0, i, j], regs_c[p])
    func = fb.func
    ir_module = IRModule(funcs={func.name: func})
    return ir_module


@pytest.mark.parametrize(
    'config',
    [
        MmaConfig.m16n8k8_f16_f16(),
        MmaConfig.m16n8k16_f16_f16(),
        MmaConfig.m16n8k8_f16_f32(),
        MmaConfig.m16n8k16_f16_f32(),
        MmaConfig.m16n8k4_tf32_f32(),
        MmaConfig.m16n8k8_tf32_f32(),
        MmaConfig.m16n8k8_bf16_f32(),
        MmaConfig.m16n8k16_bf16_f32(),
    ],
)
def test_mma(config: MmaConfig):
    if hidet.cuda.compute_capability() < (8, 0):
        if 'tf32' in [config.input_dtype, config.output_dtype]:
            pytest.skip('tfloat32 tensor core is supported on device with sm80 or higher')
        if 'bf16' in [config.input_dtype, config.output_dtype]:
            pytest.skip('bfloat16 tensor core is supported on device with sm80 or higher')
        if (config.m, config.n, config.k) in [(16, 8, 16)]:
            pytest.skip('tensor core with shape m16n8k16 is supported on device with sm80 or higher')
    ir_module = matmul_mma_tensor_core(config)
    func = build_ir_module(ir_module)
    m, n, k = config.m, config.n, config.k
    a = hidet.randint(3, shape=[1, m, k]).to(data_type(config.input_dtype).name).cuda()
    b = hidet.randint(3, shape=[1, k, n]).to(data_type(config.input_dtype).name).cuda()
    c = hidet.empty([1, m, n], dtype=data_type(config.output_dtype).name)
    func(a, b, c)
    c_desire = hidet.ops.batch_matmul(a, b)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c_desire.cpu().numpy())
