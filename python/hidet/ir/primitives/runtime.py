from typing import Union

from hidet.ir.expr import Expr
from hidet.ir.type import FuncType, uint64, boolean, void_p
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_functions():
    register_primitive_function(
        name='get_cuda_stream', func_or_type=FuncType([], void_p), codegen_name='get_cuda_stream'
    )
    register_primitive_function(
        name='request_cuda_workspace',
        func_or_type=FuncType([uint64, boolean], void_p),
        codegen_name='request_cuda_workspace',
    )
    register_primitive_function(
        name='request_cpu_workspace',
        func_or_type=FuncType([uint64, boolean], void_p),
        codegen_name='request_cpu_workspace',
    )


def get_cuda_stream() -> void_p:
    return call_primitive_func('get_cuda_stream', [])


def request_cuda_workspace(nbytes: Union[int, Expr], require_clean: Union[bool, Expr]) -> void_p:
    return call_primitive_func('request_cuda_workspace', [nbytes, require_clean])


def request_cpu_workspace(nbytes: Union[int, Expr], require_clean: Union[bool, Expr]) -> void_p:
    return call_primitive_func('request_cpu_workspace', [nbytes, require_clean])
