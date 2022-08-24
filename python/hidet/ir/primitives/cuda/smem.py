from hidet.ir.type import ScalarType
from typing import List, Optional, Union, Tuple

from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.lowlevel import PointerType, ReferenceType
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.expr import Expr, Call, cast
from hidet.ir.expr import Var
from hidet.ir.stmt import AsmStmt, BlackBoxStmt, ReturnStmt, Stmt
from hidet.ir.type import ScalarType, FuncType
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import script, attr, cast

    for dtype in ['uint8', 'uint32', 'int32', 'float16', 'float32']:
        func_name = f'cuda_dynamic_shared_memory_{dtype}'
        dtype = ScalarType(dtype)

        @script
        def cuda_dynamic_shared_memory(byte_offset: int) -> ~dtype:
            attr.func_kind = 'cuda_device'
            attr.func_name = func_name
            dynamic_smem = PointerType(base_type='uint8', specifiers=['extern', '__shared__'], use_bracket=True)
            return cast(~dynamic_smem[byte_offset], ~dtype)
        assert isinstance(cuda_dynamic_shared_memory, Function)
        register_primitive_function(cuda_dynamic_shared_memory.name, cuda_dynamic_shared_memory)


def dynamic_shared_memory(byte_offset: Union[Expr, int], dtype: Union[ScalarType, str]) -> Call:
    func_name = f'cuda_dynamic_shared_memory_{dtype}'
    return call_primitive_func(func_name, [byte_offset])


def set_kernel_max_dynamic_smem_bytes(func: Var, max_dynamic_smem_bytes: Expr) -> Stmt:
    template_string = r'cudaFuncSetAttribute({}, cudaFuncAttributeMaxDynamicSharedMemorySize, {});'
    return BlackBoxStmt(template_string, func, max_dynamic_smem_bytes)
