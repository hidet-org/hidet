from typing import List, Dict, Optional
from hidet.ir.mapping import TaskMapping, row_spatial, col_spatial, repeat_map, row_repeat, col_repeat
from hidet.utils import initialize
from hidet.ir.type import ScalarType, PointerType, VoidType, void_pointer
from hidet.ir.expr import Var, Expr, Call, cast
from hidet.ir.stmt import AsmStmt, AssignStmt, asm
from hidet.ir.builders import FunctionBuilder
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.lang import script


def resolve_cvta_func_name(src_space: str, dst_space: str) -> str:
    return 'cvta_{}_to_{}'.format(src_space, dst_space)


@initialize()
def register_cvta_instructions():
    from hidet.lang import attr, u32, tensor
    for src_space in ['generic']:
        for dst_space in ['shared']:
            if src_space == dst_space:
                continue
            func_name = 'cuda_' + resolve_cvta_func_name(src_space, dst_space)

            @script
            def cvta(src: PointerType(VoidType())) -> u32:
                attr.func_name = func_name
                ret: u32 = 0
                asm(
                    template="{.reg.u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr;}",
                    outputs=[ret],
                    inputs=[src]
                )
                return ret

            register_primitive_function(name=cvta.name, func_or_type=cvta)


def cvta_generic_to_shared(generic_addr: Expr) -> Expr:
    """
    Convert the address from generic memory space to shared memory space.

    In PTX, there are 5 memory space: generic, const, param, local, and shared. The later four are in side
    the generic memory space, and each is models as a window in generic space. The cvta (convert address)
    instructions are used to convert the address between these memory spaces.

    See Also:
    1. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#generic-addressing
    2. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta

    Parameters
    ----------
    generic_addr: Expr
        The address in generic memory space, should be a pointer.

    Returns
    -------
    ret: Expr
        The corresponding address in shared memory space. The returned address is an unsigned integer representing
        the address in shared memory space.
    """
    func_name = resolve_cvta_func_name(src_space='generic', dst_space='shared')
    return call_cuda(func_name, args=[generic_addr])

