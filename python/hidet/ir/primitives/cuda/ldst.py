from typing import Optional

from hidet.ir.dialects.lowlevel import PointerType, TensorPointerType
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
            inst_name = f'ld.{sync}.{space}.{space}.b{nbits}'
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
            inst_name = f'st.{sync}.{space}.{space}.b{nbits}'
    else:
        if space == 'generic':
            inst_name = f'st.b{nbits}'
        else:
            inst_name = f'st.{space}.b{nbits}'
    return inst_name


@initialize()
def register_functions():
    from hidet.lang import attr, script, asm

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
                    inst_name = resolve_load_inst_name(dtype, space, sync, scope)
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
    func_name = 'cuda_' + resolve_store_inst_name(dtype, space, sync, scope) + f'_{dtype}'
    return call_primitive_func(func_name, [addr, value])
