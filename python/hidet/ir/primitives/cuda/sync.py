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
from hidet.ir.type import FuncType, VoidType
from hidet.ir.primitives.func import register_primitive_function
from hidet.utils import initialize
from hidet.ir.primitives.func import call_primitive_func


@initialize()
def register_primitive_functions():
    functions = [
        ('cuda_syncthreads', '__syncthreads', FuncType([], VoidType())),
        ('cuda_syncthreads_count', '__syncthreads_count', FuncType(['int32'], 'int32')),
        ('cuda_syncthreads_and', '__syncthreads_and', FuncType(['int32'], 'int32')),
        ('cuda_syncthreads_or', '__syncthreads_or', FuncType(['int32'], 'int32')),
        ('cuda_syncwarp', '__syncwarp', FuncType([], VoidType())),
    ]
    for name, codegen_name, func_type in functions:
        register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)


def syncthreads() -> Call:
    return call_primitive_func('cuda_syncthreads', [])


def syncthreads_count(value: Expr) -> Call:
    return call_primitive_func('cuda_syncthreads_count', [value])


def syncthreads_and(cond: Union[Expr, int, bool]) -> Call:
    return call_primitive_func('cuda_syncthreads_and', [cond])


def syncthreads_or(cond: Expr) -> Call:
    return call_primitive_func('cuda_syncthreads_or', [cond])


def syncwarp() -> Call:
    return call_primitive_func('cuda_syncwarp', [])
