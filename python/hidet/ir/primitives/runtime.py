from typing import Union

from hidet.ir.expr import Expr, convert
from hidet.ir.func import Function
from hidet.ir.type import FuncType, ScalarType, uint64, uint8, boolean
from hidet.ir.dialects.lowlevel import PointerType, VoidType, void_p
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_functions():
    register_primitive_function(name='get_cuda_stream', func_or_type=FuncType([], void_p), codegen_name='get_cuda_stream')
    register_primitive_function(name='request_workspace', func_or_type=FuncType([uint64, boolean], void_p), codegen_name='request_workspace')


def get_cuda_stream() -> void_p:
    return call_primitive_func('get_cuda_stream', [])


def request_workspace(nbytes: Union[int, Expr], require_clean: Union[bool, Expr]) -> void_p:
    return call_primitive_func('request_workspace', [nbytes, require_clean])
