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

from hidet.ir.expr import Expr
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.type import FuncType, data_type
from hidet.utils import initialize


@initialize()
def register_functions():
    i32 = data_type('int32')
    register_primitive_function('cuda_atomic_add', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicAdd')
    register_primitive_function('cuda_atomic_sub', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicSub')
    register_primitive_function(
        'cuda_atomic_exchange', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicExch'
    )
    register_primitive_function(
        'cuda_atomic_cas', func_or_type=FuncType([~i32, i32, i32], i32), codegen_name='atomicCAS'
    )


def atomic_add(addr: Expr, value: Expr):
    return call_primitive_func('cuda_atomic_add', [addr, value])


def atomic_sub(addr: Expr, value: Expr):
    return call_primitive_func('cuda_atomic_sub', [addr, value])


def atomic_exchange(addr: Expr, value: Expr):
    return call_primitive_func('cuda_atomic_exchange', [addr, value])


def atomic_cas(addr: Expr, compare: Union[Expr, int], value: Union[Expr, int]):
    return call_primitive_func('cuda_atomic_cas', [addr, compare, value])
