from hidet.ir.type import ScalarType
from typing import List, Optional, Union, Tuple

from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.lowlevel import PointerType, ReferenceType
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.expr import Expr, Call, cast
from hidet.ir.expr import Var
from hidet.ir.stmt import AsmStmt, BlackBoxStmt, ReturnStmt
from hidet.ir.type import ScalarType, FuncType
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.utils import initialize


def register_unary_dialect_primitive_function(func_name, generic_func, target_dtype: str, dialect_dtype: str):
    with FunctionBuilder(func_name, kind='cuda_device', ret_type=ScalarType(target_dtype)) as fb:
        # params
        x = Var('x', type=ScalarType(target_dtype))
        fb.extend_params([x])
        # body
        sb = StmtBuilder()
        sb += ReturnStmt(cast(generic_func(cast(x, dialect_dtype)), target_dtype))
        fb.set_body(sb.finish())
    register_primitive_function(name=func_name, func_or_type=fb.get())


def register_binary_dialect_primitive_function(func_name, generic_func, target_dtype: str, dialect_dtype: str):
    with FunctionBuilder(func_name, kind='cuda_device', ret_type=ScalarType(target_dtype)) as fb:
        # params
        x = Var('x', type=ScalarType(target_dtype))
        y = Var('y', type=ScalarType(target_dtype))
        fb.extend_params([x, y])
        # body
        sb = StmtBuilder()
        sb += ReturnStmt(cast(generic_func(cast(x, dialect_dtype), cast(y, dialect_dtype)), target_dtype))
        fb.set_body(sb.finish())
    register_primitive_function(name=func_name, func_or_type=fb.get())


@initialize()
def register_primitive_functions_with_body():
    # lds128
    with FunctionBuilder('cuda_lds128', kind='cuda_device') as fb:
        # params
        regs_vars = [Var(f'reg{i}', ReferenceType(ScalarType('float32'))) for i in range(4)]
        smem_addr_var = Var('smem_addr', PointerType(ScalarType('float32')))
        fb.extend_params(regs_vars + [smem_addr_var])
        # body
        body = AsmStmt(
            r"{"
            r"  .reg.u64 u64addr;"
            r"  cvta.to.shared.u64 u64addr, %4;"
            r"  ld.shared.v4.f32 {%0, %1, %2, %3}, [u64addr];"
            r"}",
            outputs=[('=f', reg) for reg in regs_vars],
            inputs=[('l', smem_addr_var)],
            is_volatile=True
        )
        fb.set_body(body)
    register_primitive_function(name='cuda_lds128', func_or_type=fb.get())

    # sts128
    with FunctionBuilder('cuda_sts128', kind='cuda_device') as fb:
        # params
        regs_vars = [Var(f'reg{i}', ReferenceType(ScalarType('float32'))) for i in range(4)]
        smem_addr_var = Var('smem_addr', PointerType(ScalarType('float32')))
        fb.extend_params(regs_vars + [smem_addr_var])
        # body
        body = AsmStmt(
            r"{"
            r"  .reg.u64 u64addr;"
            r"  cvta.to.shared.u64 u64addr, %0;"
            r"  st.shared.v4.f32 [u64addr], {%1, %2, %3, %4};"
            r"}",
            outputs=[],
            inputs=[('l', smem_addr_var)] + [('f', reg) for reg in regs_vars],
            is_volatile=True
        )
        fb.set_body(body)
    register_primitive_function(name='cuda_sts128', func_or_type=fb.get())


@initialize()
def register_primitive_functions():
    functions = [
        ('cuda_syncthreads', '__syncthreads', FuncType([], VoidType())),
        ('cuda_syncwarp', '__syncwarp', FuncType([], VoidType())),
        ('cuda_activemask', '__activemask', FuncType([], 'int32')),
        ('cuda_shfl_sync', '__shfl_sync', FuncType(type_infer_func=lambda arg_types: arg_types[1])),    # T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)
        ('cuda_shfl_up_sync', '__shfl_up_sync', FuncType(type_infer_func=lambda arg_types: arg_types[1])),
        ('cuda_shfl_down_sync', '__shfl_down_sync', FuncType(type_infer_func=lambda arg_types: arg_types[1])),
    ]
    # functions = {
    #     'cuda_syncthreads'
    #     '__syncthreads': FuncType([], VoidType()),
    #     '__syncwarp': FuncType([], VoidType()),
    #     '__activemask': FuncType([], 'int32'),
    #     '__shfl_sync': FuncType(type_infer_func=lambda arg_types: arg_types[1]),    # T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)
    #     '__shfl_up_sync': FuncType(type_infer_func=lambda arg_types: arg_types[1]),
    #     '__shfl_down_sync': FuncType(type_infer_func=lambda arg_types: arg_types[1]),
    # }
    for name, codegen_name, func_type in functions:
        register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)


def call_cuda(func_name, args: List[Expr]) -> Call:
    entry = primitive_func_pool.lookup_by_name('cuda_{}'.format(func_name))
    return Call(entry.var, args)


def syncthreads() -> Call:
    return call_cuda('syncthreads', [])


def syncwarp() -> Call:
    return call_cuda('syncwarp', [])


def lds128(reg0, reg1, reg2, reg3, smem_addr) -> Call:
    return call_cuda('lds128', [reg0, reg1, reg2, reg3, smem_addr])


def sts128(reg0, reg1, reg2, reg3, smem_addr) -> Call:
    return call_cuda('sts128', [reg0, reg1, reg2, reg3, smem_addr])


def shfl_sync(mask, var, src_lane, width=32):
    return call_cuda('shfl_sync', [mask, var, src_lane, width])


def shfl_up_sync(mask, var, delta, width=32):
    return call_cuda('shfl_up_sync', [mask, var, delta, width])


def shfl_down_sync(mask, var, delta, width=32):
    return call_cuda('shfl_down_sync', [mask, var, delta, width])


def shfl_xor_sync(mask, var, lane_mask, width=32):
    return call_cuda('shfl_down_sync', [mask, var, lane_mask, width])


def active_mask():
    return call_cuda('activemask', [])


def set_kernel_max_dynamic_smem_bytes(func, max_dynamic_smem_bytes):
    template_string = r'cudaFuncSetAttribute({}, cudaFuncAttributeMaxDynamicSharedMemorySize, {});'
    raise ValueError('update to use func instead of func_var')
    return BlackBoxStmt(template_string, func, max_dynamic_smem_bytes)
