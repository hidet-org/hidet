import numpy as np
import pytest

import hidet
from hidet.driver import build_ir_module
from hidet.ir.builders import FunctionBuilder
from hidet.ir.dialects.lowlevel import TensorPointerType
from hidet.ir.expr import Var, tensor_var
from hidet.ir.func import IRModule
from hidet.ir.primitives.cuda import thread_idx
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_sync, mma_configs
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.type import ScalarType


def matmul_mma_tensor_core(config: MmaConfig):
    with FunctionBuilder(
            name='matmul_mma_grid',
            kind='cuda_kernel',
            grid_dim=1,
            block_dim=32
    ) as fb:
        # parameters
        a = Var('a', TensorPointerType('global', config.input_dtype, [config.m, config.k]))
        b = Var('b', TensorPointerType('global', config.input_dtype, [config.k, config.n]))
        c = Var('c', TensorPointerType('global', config.output_dtype, [config.m, config.n]))
        fb.extend_params([a, b, c])

        # local variables
        regs_a = tensor_var('regs_a', [config.a_elements], 'register', config.input_dtype)
        regs_b = tensor_var('regs_b', [config.b_elements], 'register', config.input_dtype)
        regs_c = tensor_var('regs_c', [config.c_elements], 'register', config.output_dtype)
        fb.extend_local_vars([regs_a, regs_b, regs_c])

        # body
        w = thread_idx()
        for p in range(config.c_elements):
            fb += BufferStoreStmt(regs_c, [p], 0.0)
        for p, (i, k) in enumerate(config.a_load_map(w)):
            fb += BufferStoreStmt(regs_a, [p], a[i, k])
        for p, (k, j) in enumerate(config.b_load_map(w)):
            fb += BufferStoreStmt(regs_b, [p], b[k, j])
        fb += mma_sync(config, regs_a, regs_b, regs_c)
        for p, (i, j) in enumerate(config.c_store_map(w)):
            fb += BufferStoreStmt(c, [i, j], regs_c[p])
    func = fb.func
    return IRModule(funcs={func.name: func})


@pytest.mark.parametrize(
    'config', [
        MmaConfig.m16n8k8_f16_f16(),
        MmaConfig.m16n8k16_f16_f16(),
        MmaConfig.m16n8k8_f16_f32(),
        MmaConfig.m16n8k16_f16_f32(),
        MmaConfig.m16n8k4_tf32_f32(),
        MmaConfig.m16n8k8_tf32_f32(),
        MmaConfig.m16n8k8_bf16_f32(),
        MmaConfig.m16n8k16_bf16_f32(),
    ]
)
def test_mma(config: MmaConfig):
    ir_module = matmul_mma_tensor_core(config)
    func = build_ir_module(ir_module, func_name='matmul_mma', keep_ptx=True)
    m, n, k = config.m, config.n, config.k
    a = hidet.randint(3, shape=[m, k]).to(ScalarType(config.input_dtype).name)
    b = hidet.randint(3, shape=[k, n]).to(ScalarType(config.input_dtype).name)
    c = hidet.empty([m, n], dtype=ScalarType(config.output_dtype).name)
    func(a, b, c)
    c_desire = hidet.ops.matmul(a, b)
    np.testing.assert_allclose(actual=c.numpy(), desired=c_desire.numpy())
