from typing import List, Dict, Optional

from hidet.ir.mapping import TaskMapping, row_spatial, col_spatial, repeat_map, row_repeat, col_repeat
from hidet.utils import initialize
from hidet.ir.type import ScalarType
from hidet.ir.expr import Var, Expr, cast
from hidet.ir.stmt import AsmStmt, AssignStmt, asm
from hidet.ir.func import Function
from hidet.ir.dialects.lowlevel import PointerType, VoidType
from hidet.ir.builders import FunctionBuilder
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda


def resolve_load_name(dtype: ScalarType, space: str, sync: Optional[str], scope: str) -> str:
    sync_field = f'_{sync}' if sync else '',
    return f'load_{space}_{dtype.name}{sync_field}_{scope}'


def resolve_store_name(dtype: ScalarType, space: str, sync: Optional[str], scope: str):
    sync_field = f'_{sync}' if sync else '',
    return f'store_{space}_{dtype.name}{sync_field}_{scope}'


@initialize()
def register_functions():
    from hidet.lang import script, i32

    @script
    def cuda_lock(addr: ~i32):
        while True:
            pass

    @script
    def cuda_unlock(addr: ~i32):
        pass


def load(addr: Expr, space: str = 'generic', sync: Optional[str] = 'acquire', scope: str = 'gpu'):
    """
    Load data from memory.

    Parameters
    ----------
    addr: Expr
        The address of the data, in a type of pointer.

    space: str
        The memory space of the address. Candidates: 'generic', 'global', 'shared', 'local'

    sync: Optional[str]
        The synchronization behavior, can be None, 'acquire', and 'relaxed'.

    scope: str
        The scope of the synchronization.

    Returns
    -------
    ret: Expr
        The loaded data.
    """
    pass


def store(addr: Expr, space: str = 'generic', sync: Optional[str] = 'release', scope: str = 'gpu'):
    """
    Store data to memory.

    Parameters
    ----------
    addr: Expr
        The address to store the data.

    space: str
        The memory space of the address. Candidates: 'generic', 'global', 'shared', 'local'

    sync: Optional[str]
        The synchronization behavior, can be None, 'release', and 'relaxed'.

    scope: str
        The scope of the synchronization.
    """
    pass
