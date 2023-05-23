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
from typing import Dict

from hidet.ir.expr import Var
from hidet.ir.type import DataType


registered_primitive_variables: Dict[str, Var] = {}


def register_primitive_variable(name: str, dtype: DataType):
    if name in registered_primitive_variables:
        raise KeyError('Primitive variable {} has already registered.'.format(name))
    var = Var(hint=None, type=dtype, name=name)
    registered_primitive_variables[name] = var
    return var


def lookup_primitive_variable(name: str) -> Var:
    if name not in registered_primitive_variables:
        raise KeyError('Primitive variable {} has not registered.'.format(name))
    return registered_primitive_variables[name]
