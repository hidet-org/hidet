from typing import Dict

from hidet.ir.expr import Var
from hidet.ir.type import ScalarType

_primitive_variables: Dict[str, Var] = {}


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
