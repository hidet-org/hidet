from typing import Dict, Callable, Set, Union, Optional, Tuple
from hidet.ir.type import FuncType, ScalarType
from hidet.ir.expr import Var, Call
from hidet.ir.stmt import AsmStmt, BlackBoxStmt
from hidet.ir.func import Function
from hidet.ir.task import Thread
from hidet.ir.dialects.lowlevel import VoidType, PointerType, ReferenceType
from hidet.ir.builders import FunctionBuilder

_primitive_functions: Dict[str, Tuple[Var, FuncType, Optional[Function]]] = {}


def is_primitive_function(name):
    return name in _primitive_functions


def get_primitive_function(name: str) -> Tuple[Var, FuncType, Optional[Function]]:
    assert name in _primitive_functions
    return _primitive_functions[name]


def register_primitive_function(name, func_or_ftype: Optional[Union[Function, FuncType]] = None):
    if isinstance(func_or_ftype, Function):
        func = func_or_ftype
        func_type = FuncType.from_func(func)
    elif isinstance(func_or_ftype, FuncType):
        func = None
        func_type = func_or_ftype
    elif func_or_ftype is None:
        # currently, we allow to register primitive function without function type
        # leave the check to underlying compiler
        # todo: add the check at our level
        func = None
        func_type = None
    else:
        raise False
    v = Var(name, func_type)
    assert name not in _primitive_functions
    _primitive_functions[name] = (v, func_type, func)


def syncthreads() -> Call:
    if '__syncthreads' not in _primitive_functions:
        register_primitive_function('__syncthreads', FuncType([], VoidType()))
    func_var = get_primitive_function('__syncthreads')[0]
    return Call(func_var, [])


def lds128(reg0, reg1, reg2, reg3, smem_addr) -> Call:
    if 'lds128' not in _primitive_functions:
        with FunctionBuilder('lds128', attrs={'worker': Thread()}) as fb:
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
        register_primitive_function('lds128', fb.get())
    func_var = get_primitive_function('lds128')[0]
    return Call(func_var, [reg0, reg1, reg2, reg3, smem_addr])


def sts128(reg0, reg1, reg2, reg3, smem_addr) -> Call:
    if 'sts128' not in _primitive_functions:
        with FunctionBuilder('sts128', attrs={'worker': Thread()}) as fb:
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
        register_primitive_function('sts128', fb.get())
    func_var = get_primitive_function('sts128')[0]
    return Call(func_var, [reg0, reg1, reg2, reg3, smem_addr])


def printf(format_string, *args):
    """
    usage:
    printf(r"%d %d\n", expr_1, expr_2)
    """
    arg_string = ', '.join(['{}'] * len(args))
    template_string = f'printf("{format_string}", {arg_string});'
    return BlackBoxStmt(template_string, *args)


def shfl_sync(mask, var, src_lane, width=32):
    if '__shfl_sync' not in _primitive_functions:
        register_primitive_function('__shfl_sync', FuncType([], VoidType()))
    func_var = get_primitive_function('__shfl_sync')[0]
    return Call(func_var, [mask, var, src_lane, width])


def shfl_up_sync(mask, var, delta, width=32):
    if '__shfl_up_sync' not in _primitive_functions:
        register_primitive_function('__shfl_up_sync')
    func_var = get_primitive_function('__shfl_up_sync')[0]
    return Call(func_var, [mask, var, delta, width])


def shfl_down_sync(mask, var, delta, width=32):
    if '__shfl_down_sync' not in _primitive_functions:
        register_primitive_function('__shfl_down_sync')
    func_var = get_primitive_function('__shfl_down_sync')[0]
    return Call(func_var, [mask, var, delta, width])


def shfl_xor_sync(mask, var, lane_mask, width=32):
    if '__shfl_down_sync' not in _primitive_functions:
        register_primitive_function('__shfl_down_sync')
    func_var = get_primitive_function('__shfl_down_sync')[0]
    return Call(func_var, [mask, var, lane_mask, width])


def expf(v):
    if '__expf' not in _primitive_functions:
        register_primitive_function('__expf')
    func_var = get_primitive_function('__expf')[0]
    return Call(func_var, [v])


def active_mask():
    if '__activemask' not in _primitive_functions:
        register_primitive_function('__activemask')
    func_var = get_primitive_function('__activemask')[0]
    return Call(func_var, [])


def cuda_max(a, b):
    if 'max' not in _primitive_functions:
        register_primitive_function('max')
    func_var = get_primitive_function('max')[0]
    return Call(func_var, [a, b])


def cuda_min(a, b):
    if 'min' not in _primitive_functions:
        register_primitive_function('min')
    func_var = get_primitive_function('min')[0]
    return Call(func_var, [a, b])


def set_kernel_max_dynamic_smem_bytes(func_var, max_dynamic_smem_bytes):
    template_string = r'cudaFuncSetAttribute({}, cudaFuncAttributeMaxDynamicSharedMemorySize, {});'
    return BlackBoxStmt(template_string, func_var, max_dynamic_smem_bytes)
