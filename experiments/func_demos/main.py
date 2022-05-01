import numpy as np
import hidet
from hidet.ir.func import IRModule
from hidet.ir.builders import FunctionBuilder
from hidet.ir.dialects.lowlevel import PointerType
from hidet.ir.type import ScalarType, TensorType
from hidet.ir.expr import Var, tensor_var
from hidet.ir.primitives.cuda.wmma import wmma_load_a, wmma_load_b, wmma_store, wmma_mma
from hidet.driver import build_ir_module


def matmul_tensor_core() -> IRModule:
    dtype = 'float16'
    with FunctionBuilder(
            name='matmul_tensor_core_grid',
            kind='cuda_kernel',
            grid_dim=1,
            block_dim=32
    ) as fb:
        # parameters: a, b, c
        a = Var('a', PointerType(ScalarType(dtype)))
        b = Var('b', PointerType(ScalarType(dtype)))
        c = Var('c', PointerType(ScalarType(dtype)))
        fb.extend_params([a, b, c])

        # local variables
        regs_a = tensor_var('regs_a', [8], scope='register', dtype='uint32')
        regs_b = tensor_var('regs_b', [8], scope='register', dtype='uint32')
        regs_c = tensor_var('regs_c', [4], scope='register', dtype='uint32')
        fb.extend_local_vars([regs_a, regs_b, regs_c])

        # body
        fb += wmma_load_a(shape=(16, 16, 16), dtype=dtype, reg_addr=regs_a, mem_addr=a)
        fb += wmma_load_b(shape=(16, 16, 16), dtype=dtype, reg_addr=regs_b, mem_addr=b)
        fb += wmma_mma(shape=(16, 16, 16), a_dtype=dtype, b_dtype=dtype, c_dtype=dtype, a_regs_addr=regs_a, b_regs_addr=regs_b, c_regs_addr=regs_c)
        fb += wmma_store(shape=(16, 16, 16), dtype=dtype, mem_addr=c, reg_addr=regs_c)
    func = fb.func
    return IRModule(funcs={func.name: func})


def main():
    ir_module = matmul_tensor_core()
    func = build_ir_module(ir_module, func_name='matmul_tensor_core', keep_ptx=True)
    na = np.arange(16 * 16).astype(np.float16).reshape([16, 16]) / 100.0
    nb = np.arange(16 * 16).astype(np.float16).reshape([16, 16]) / 100.0

    a = hidet.array(na)
    b = hidet.array(nb)
    c = hidet.empty(shape=[16, 16], dtype='float16')
    func(a, b, c)
    print(a)
    print(b)
    print(c)
    c2 = hidet.ops.matmul(a, b)
    print(c2 - c)


if __name__ == '__main__':
    main()
