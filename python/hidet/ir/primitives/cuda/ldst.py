# pylint: disable=cell-var-from-loop
from typing import Optional

from hidet.ir.type import PointerType, TensorPointerType
from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.functors import infer_type
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.type import ScalarType
from hidet.utils import initialize


def resolve_load_inst_name(dtype: str, space: str, sync: Optional[str], scope: str) -> str:
    dtype = ScalarType(dtype)
    nbytes = dtype.nbytes()
    nbits = nbytes * 8
    if sync:
        if space == 'generic':
            inst_name = f'ld.{sync}.{scope}.b{nbits}'
        else:
            inst_name = f'ld.{sync}.{scope}.{space}.b{nbits}'
    else:
        if space == 'generic':
            inst_name = f'ld.b{nbits}'
        else:
            inst_name = f'ld.{space}.b{nbits}'
    return inst_name


def resolve_store_inst_name(dtype: str, space: str, sync: Optional[str], scope: str) -> str:
    dtype = ScalarType(dtype)
    nbytes = dtype.nbytes()
    nbits = nbytes * 8
    if sync:
        if space == 'generic':
            inst_name = f'st.{sync}.{scope}.b{nbits}'
        else:
            inst_name = f'st.{sync}.{scope}.{space}.b{nbits}'
    else:
        if space == 'generic':
            inst_name = f'st.b{nbits}'
        else:
            inst_name = f'st.{space}.b{nbits}'
    return inst_name


@initialize()
def register_functions():
    from hidet.lang import attr, script, asm  # pylint: disable=import-outside-toplevel

    registered = set()
    for dtype in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int32', 'int64', 'float16', 'float32']:
        for space in ['generic', 'global', 'local', 'shared']:
            for sync in [None, 'acquire', 'relaxed']:
                for scope in ['cta', 'gpu', 'sys']:
                    inst_name = resolve_load_inst_name(dtype, space, sync, scope)
                    func_name = 'cuda_' + inst_name.replace('.', '_') + f'_{dtype}'
                    if func_name in registered:
                        continue
                    registered.add(func_name)

                    @script
                    def cuda_load(addr: ~ScalarType(dtype)) -> ScalarType(dtype):
                        attr.func_kind = 'cuda_device'
                        attr.func_name = func_name
                        template = inst_name + ' %0, [%1];'
                        ret: ScalarType(dtype) = 0  # define a variable used to store the loaded data
                        asm(template, outputs=[ret], inputs=[addr], is_volatile=True)
                        return ret

                    assert isinstance(cuda_load, Function)
                    register_primitive_function(name=cuda_load.name, func_or_type=cuda_load)

    for dtype in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int32', 'int64', 'float16', 'float32']:
        for space in ['generic', 'global', 'local', 'shared']:
            for sync in [None, 'release', 'relaxed']:
                for scope in ['cta', 'gpu', 'sys']:
                    inst_name = resolve_store_inst_name(dtype, space, sync, scope)
                    func_name = 'cuda_' + inst_name.replace('.', '_') + f'_{dtype}'
                    if func_name in registered:
                        continue
                    registered.add(func_name)

                    @script
                    def cuda_store(addr: ~ScalarType(dtype), value: ScalarType(dtype)):
                        attr.func_kind = 'cuda_device'
                        attr.func_name = func_name
                        template = inst_name + ' [%0], %1;'
                        asm(template, inputs=[addr, value], is_volatile=True)

                    assert isinstance(cuda_store, Function)
                    register_primitive_function(name=cuda_store.name, func_or_type=cuda_store)


@initialize()
def register_primitive_functions_with_body():
    # pylint: disable=import-outside-toplevel
    from hidet.ir.type import ReferenceType
    from hidet.ir.expr import Var
    from hidet.ir.stmt import AsmStmt
    from hidet.ir.builders import FunctionBuilder

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
            is_volatile=True,
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
            is_volatile=True,
        )
        fb.set_body(body)
    register_primitive_function(name='cuda_sts128', func_or_type=fb.get())


def resolve_pointed_dtype(addr: Expr) -> str:
    ptr_type = infer_type(addr)
    if not isinstance(ptr_type, (PointerType, TensorPointerType)):
        raise ValueError('Expect a pointer type, got {}'.format(ptr_type))
    if isinstance(ptr_type, PointerType):
        dtype = ptr_type.base_type
    else:
        dtype = ptr_type.tensor_type.scalar_type
    if not isinstance(dtype, ScalarType):
        raise ValueError('Expect a pointer to a scalar type, got {}'.format(ptr_type))
    return dtype.name


def load(addr: Expr, space: str = 'generic', sync: Optional[str] = None, scope: Optional[str] = None):
    """
    Load data from memory.

    Parameters
    ----------
    addr: Expr
        The address of the data, in a type of pointer.

    space: str
        The memory space of the address. Candidates: 'generic', 'global', 'shared', 'local'

    sync: Optional[str]
        The synchronization behavior. Candidates: None, 'acquire', and 'relaxed'.

    scope: Optional[str]
        The scope of the synchronization. Candidates: None, 'cta', 'gpu', 'sys'.

    Returns
    -------
    ret: Expr
        The loaded data.
    """
    dtype = resolve_pointed_dtype(addr)
    func_name = 'cuda_' + resolve_load_inst_name(dtype, space, sync, scope).replace('.', '_') + f'_{dtype}'
    return call_primitive_func(func_name, [addr])


def store(addr: Expr, value: Expr, space: str = 'generic', sync: Optional[str] = 'release', scope: str = 'gpu'):
    """
    Store data to memory.

    Parameters
    ----------
    addr: Expr
        The address to store the data.

    value: Expr
        The value to store.

    space: str
        The memory space of the address. Candidates: 'generic', 'global', 'shared', 'local'

    sync: Optional[str]
        The synchronization behavior. Candidates: 'release', and 'relaxed'.

    scope: str
        The scope of the synchronization. Candidates: 'cta', 'gpu', 'sys'.
    """
    dtype = resolve_pointed_dtype(addr)
    func_name = 'cuda_' + resolve_store_inst_name(dtype, space, sync, scope).replace('.', '_') + f'_{dtype}'
    return call_primitive_func(func_name, [addr, value])


def lds128(reg0, reg1, reg2, reg3, smem_addr):
    return call_primitive_func('cuda_lds128', [reg0, reg1, reg2, reg3, smem_addr])


def sts128(reg0, reg1, reg2, reg3, smem_addr):
    return call_primitive_func('cuda_sts128', [reg0, reg1, reg2, reg3, smem_addr])
