import numpy
import numpy as np
import hidet
from hidet.ir.func import IRModule
from hidet.ir.builders import FunctionBuilder
from hidet.ir.dialects.lowlevel import PointerType, TensorPointerType
from hidet.ir.type import ScalarType, TensorType
from hidet.ir.expr import Var, tensor_var, cast
from hidet.ir.stmt import BufferStoreStmt
from hidet.tos.ops.definitions.matmul.matmul import MatmulTask, input_like
from hidet.ir.primitives.cuda.wmma import wmma_load_a, wmma_load_b, wmma_store, wmma_mma, WmmaConfig
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_sync, mma_configs
from hidet.ir.primitives.cuda import thread_idx
from hidet.tos.ops.schedules.cuda.matmul.mma import batched_matmul_cuda_schedule_mma, batched_matmul_cuda_with_given_schedule, MatmulMmaSchedule
from hidet.ir.primitives import printf
from hidet.driver import build_ir_module
from hidet.ir.mapping import row_repeat, col_repeat


def matmul_wmma_tensor_core() -> IRModule:
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
        config = WmmaConfig(
            shape=(16, 16, 16),
            a_dtype='f16',
            b_dtype='f16',
            c_dtype='f16',
            a_layout='row',
            b_layout='row',
            c_layout='row',
            a_regs=8,
            b_regs=8,
            c_regs=4
        )
        fb += wmma_load_a(config=config, reg_addr=regs_a, mem_addr=a)
        fb += wmma_load_b(config=config, reg_addr=regs_b, mem_addr=b)
        fb += wmma_mma(config=config, a_regs_addr=regs_a, b_regs_addr=regs_b, c_regs_addr=regs_c)
        fb += wmma_store(config=config, mem_addr=c, reg_addr=regs_c)
    func = fb.func
    return IRModule(funcs={func.name: func})


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


def run_config(config: MmaConfig):
    ir_module = matmul_mma_tensor_core(config)
    func = build_ir_module(ir_module, func_name='matmul_mma', keep_ptx=True)
    m, n, k = config.m, config.n, config.k
    a = hidet.randint(3, shape=[m, k]).to(ScalarType(config.input_dtype).name)
    b = hidet.randint(3, shape=[k, n]).to(ScalarType(config.input_dtype).name)
    c = hidet.empty([m, n], dtype=ScalarType(config.output_dtype).name)
    func(a, b, c)
    c_desire = hidet.ops.matmul(a, b)
    np.testing.assert_allclose(actual=c.numpy(), desired=c_desire.numpy())


def demo_matmul_mma():
    for config in mma_configs.values():
        # config = MmaConfig.m16n8k8_f16_f16()
        run_config(config)
    # run_config(MmaConfig.m16n8k16_f16_f16())
    # run_config(MmaConfig.m16n8k4_tf32_f32())


def main():
    ir_module = matmul_wmma_tensor_core()
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


def demo_mma():
    m, n, k = 997, 1001, 89
    a = hidet.randint(2, shape=[1, m, k], dtype='float16')
    b = hidet.randint(2, shape=[1, k, n], dtype='float16')
    c = hidet.empty([1, m, n], dtype='float16')
    task = MatmulTask(input_like(a, 'a'), input_like(b, 'b'), mma='default')
    sch = MatmulMmaSchedule(
        block_shape=[16, 8, 32],
        warp_shape=[16, 8, 16],
        mma_config=MmaConfig.m16n8k8_f16_f16()
    )
    print(sch)
    ir_module = batched_matmul_cuda_with_given_schedule(
        task=task,
        sch=sch
    )
    # # print(ir_module)
    func = build_ir_module(ir_module, func_name=task.name, keep_ptx=True)
    func(a, b, c)
    hidet.utils.cuda.device_synchronize()
    c_desired = hidet.ops.matmul(a, b)
    # print(a)
    # print(b)
    # print(c)
    # print(c_desired)
    np.testing.assert_allclose(c.numpy(), c_desired.numpy())


def demo_mma_op():
    mn = 4777
    m, n, k = mn, mn, 777
    # a = hidet.randint(2, shape=[1, m, k], dtype='float16')
    # b = hidet.randint(2, shape=[1, k, n], dtype='float16')
    a = hidet.ones(shape=[1, m, k], dtype='float16')
    b = hidet.ones(shape=[1, k, n], dtype='float16')
    numpy.set_printoptions(linewidth=200)
    with hidet.utils.CacheDir('./outs/cache'):
        hidet.utils.hidet_clear_op_cache()
        for mma_type in [
            'mma_f16_f16',
            # 'mma_f16_f32',
            # 'mma_bf16_f32',
            # 'mma_tf32_f32'
        ]:
            hidet.space_level(1)
            c1 = hidet.ops.matmul(a, b, mma=mma_type)
            c2 = hidet.ops.matmul(a, b)
            # print(c1)
            # print(c2)
            # print(c1 - c2)
        np.testing.assert_allclose(c1.numpy(), c2.numpy())


if __name__ == '__main__':
    # main()
    # matmul_mma_tensor_core()
    # demo_matmul_mma()
    # print(row_repeat(2, 2))
    # for (i, j) in col_repeat(2, 2)(0):
    #     print(i, j)
    # demo_mma()
    demo_mma_op()

