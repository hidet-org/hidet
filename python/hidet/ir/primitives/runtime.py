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
from hidet.ir.type import FuncType, void_p, string_type
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.dtypes import int64, boolean, int32
from hidet.utils import initialize


@initialize()
def register_functions():
    register_primitive_function(
        name='get_cuda_stream', func_or_type=FuncType([], void_p), codegen_name='get_cuda_stream'
    )
    register_primitive_function(
        name='request_cuda_workspace',
        func_or_type=FuncType([int64, boolean], void_p),
        codegen_name='request_cuda_workspace',
    )
    register_primitive_function(
        name='request_cpu_workspace',
        func_or_type=FuncType([int64, boolean], void_p),
        codegen_name='request_cpu_workspace',
    )
    register_primitive_function(
        name='get_symbol_value', func_or_type=FuncType([string_type()], int32), codegen_name='get_symbol_value'
    )
    register_primitive_function(
        name='set_symbol_value', func_or_type=FuncType([string_type(), int32], void_p), codegen_name='set_symbol_value'
    )
    register_primitive_function(
        name='memory_planner_init', func_or_type=FuncType([], void_p), codegen_name='memory_planner_init'
    )
    register_primitive_function(
        name='memory_planner_allocate', func_or_type=FuncType([int64], void_p), codegen_name='memory_planner_allocate'
    )
    register_primitive_function(
        name='memory_planner_free', func_or_type=FuncType([int64], void_p), codegen_name='memory_planner_free'
    )
    register_primitive_function(
        name='memory_planner_used', func_or_type=FuncType([], int64), codegen_name='memory_planner_used'
    )


def get_cuda_stream() -> void_p:
    return call_primitive_func('get_cuda_stream', [])


def request_cuda_workspace(nbytes: Union[int, Expr], require_clean: Union[bool, Expr]) -> void_p:
    return call_primitive_func('request_cuda_workspace', [nbytes, require_clean])


def request_cpu_workspace(nbytes: Union[int, Expr], require_clean: Union[bool, Expr]) -> void_p:
    return call_primitive_func('request_cpu_workspace', [nbytes, require_clean])


def get_symbol_value(name: Union[str, Expr]) -> int32:
    return call_primitive_func('get_symbol_value', [name])


def set_symbol_value(name: Union[str, Expr], value: Union[int, Expr]):
    return call_primitive_func('set_symbol_value', [name, value])


def memory_planner_init():
    return call_primitive_func('memory_planner_init', [])


def memory_planner_allocate(size: Union[int, Expr]):
    return call_primitive_func('memory_planner_allocate', [size])


def memory_planner_free(ptr: Union[int, Expr]):
    return call_primitive_func('memory_planner_free', [ptr])


def memory_planner_used():
    return call_primitive_func('memory_planner_used', [])
