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
from typing import Union

from hidet.ir.expr import Expr, Call
from hidet.ir.type import FuncType, VoidType, PointerType
from hidet.ir.primitives.func import register_primitive_function
from hidet.utils import initialize
from hidet.ir.primitives.func import call_primitive_func


@initialize()
def register_primitive_functions():
    functions = [
        ('cpu_atomic_load_n', '__atomic_load_n', FuncType([PointerType(VoidType()), 'int32'], 'int32')),
        ('cpu_atomic_add_fetch', '__atomic_add_fetch', FuncType([PointerType(VoidType()), 'int32', 'int32'], 'int32')),
        ('cpu_atomic_fetch_xor', '__atomic_fetch_xor', FuncType([PointerType(VoidType()), 'int32', 'int32'], 'int32')),
    ]

    for name, codegen_name, func_type in functions:
        register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)


def cpu_atomic_load_n(ptr: Expr, order: Union[Expr, int]) -> Expr:
    return call_primitive_func('cpu_atomic_load_n', [ptr, order])


def cpu_atomic_add_fetch(ptr: Expr, val: Union[Expr, int], order: Union[Expr, int]) -> Expr:
    return call_primitive_func('cpu_atomic_add_fetch', [ptr, val, order])


def cpu_atomic_fetch_xor(ptr: Expr, val: Union[Expr, int], order: Union[Expr, int]) -> Expr:
    return call_primitive_func('cpu_atomic_fetch_xor', [ptr, val, order])

















