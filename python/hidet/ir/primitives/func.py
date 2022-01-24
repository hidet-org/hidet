from typing import Dict, Callable, Set, Union, Optional, Tuple
from hidet.ir.expr import Var, Call
from hidet.ir.type import FuncType, ScalarType
from hidet.ir.func import Function
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.builders import FunctionBuilder

_primitive_functions: Dict[str, Tuple[Var, FuncType, Optional[Function]]] = {}
_primitive_variables: Dict[str, Var] = {}


def is_primitive_function(name):
    return name in _primitive_functions


def get_primitive_function(name: str) -> Tuple[Var, FuncType, Optional[Function]]:
    assert name in _primitive_functions
    return _primitive_functions[name]


def register_primitive_function(name, func_or_ftype: Union[Function, FuncType]):
    if isinstance(func_or_ftype, Function):
        func = func_or_ftype
        func_type = FuncType.from_func(func)
    elif isinstance(func_or_ftype, FuncType):
        func = None
        func_type = func_or_ftype
    else:
        raise False
    v = Var(name, func_type)
    assert name not in _primitive_functions
    _primitive_functions[name] = (v, func_type, func)


register_primitive_function('__syncthreads', FuncType([], VoidType()))


def syncthreads() -> Call:
    func_var = get_primitive_function('__syncthreads')[0]
    return Call(func_var, [])


def thread_idx() -> Var:
    name = 'threadIdx.x'
    if name not in _primitive_variables:
        _primitive_variables[name] = Var(name, ScalarType('int32'))
    return _primitive_variables[name]


def block_idx() -> Var:
    name = 'blockIdx.x'
    if name not in _primitive_variables:
        _primitive_variables[name] = Var(name, ScalarType('int32'))
    return _primitive_variables[name]
