# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import namedtuple
from typing import Dict, Optional, List

from hidet.ir.expr import Var
from hidet.ir.type import data_type

_primitive_variables: Dict[str, Var] = {}


def attach_pool(var):
    if '_primitive_variables' not in var.__dict__:
        var.__dict__['_primitive_variables'] = _primitive_variables
    return var


def thread_idx(dim='x') -> Var:
    assert dim in ['x', 'y', 'z']
    name = 'threadIdx.{}'.format(dim)
    if name not in _primitive_variables:
        _primitive_variables[name] = attach_pool(Var(hint=name, type=data_type('int32'), name=name))
    return _primitive_variables[name]


def block_idx(dim='x') -> Var:
    assert dim in ['x', 'y', 'z']
    name = 'blockIdx.{}'.format(dim)
    if name not in _primitive_variables:
        _primitive_variables[name] = attach_pool(Var(hint=name, type=data_type('int32'), name=name))
    return _primitive_variables[name]


def block_dim(dim='x') -> Var:
    assert dim in ['x', 'y', 'z']
    name = 'blockIdx.{}'.format(dim)
    if name not in _primitive_variables:
        _primitive_variables[name] = attach_pool(Var(hint=name, type=data_type('int32'), name=name))
    return _primitive_variables[name]


def grid_dim(dim='x') -> Var:
    assert dim in ['x', 'y', 'z']
    name = 'gridIdx.{}'.format(dim)
    if name not in _primitive_variables:
        _primitive_variables[name] = attach_pool(Var(hint=name, type=data_type('int32'), name=name))
    return _primitive_variables[name]


def is_primitive_variable(name: str) -> bool:
    return name in _primitive_variables


def get_primitive_variable(name: str) -> Optional[Var]:
    if name in _primitive_variables:
        return _primitive_variables[name]
    else:
        return None


def get_all_primitive_vars() -> List[Var]:
    return list(_primitive_variables.values())


dim3 = namedtuple('dim3', field_names=['x', 'y', 'z'])
threadIdx = dim3(thread_idx('x'), thread_idx('y'), thread_idx('z'))
blockIdx = dim3(block_idx('x'), block_idx('y'), block_idx('z'))
blockDim = dim3(block_dim('x'), block_dim('y'), block_dim('z'))
gridDim = dim3(grid_dim('x'), grid_dim('y'), grid_dim('z'))
