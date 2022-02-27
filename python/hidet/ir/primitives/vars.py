from typing import Dict, Optional

from hidet.ir.expr import Var
from hidet.ir.type import ScalarType

_primitive_variables: Dict[str, Var] = {}


def thread_idx() -> Var:
    name = 'threadIdx.x'
    if name not in _primitive_variables:
        _primitive_variables[name] = Var(hint=name, type=ScalarType('int32'), name=name)
    return _primitive_variables[name]


def block_idx() -> Var:
    name = 'blockIdx.x'
    if name not in _primitive_variables:
        _primitive_variables[name] = Var(hint=name, type=ScalarType('int32'), name=name)
    return _primitive_variables[name]

def is_primitive_variable(name: str) -> bool:
    return name in _primitive_variables

def get_primitive_variable(name: str) -> Optional[Var]:
    if name in _primitive_variables:
        return _primitive_variables[name]
    else:
        return None
