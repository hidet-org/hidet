from typing import List, Dict

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


def lock(addr: Expr, scope: str = 'gpu'):
    pass


def unlock(addr: Expr, scope: str = 'gpu'):
    pass

