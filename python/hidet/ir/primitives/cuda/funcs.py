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
from typing import List

from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import Expr, Call, cast
from hidet.ir.expr import Var
from hidet.ir.stmt import ReturnStmt
from hidet.ir.type import data_type
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool


def register_unary_dialect_primitive_function(func_name, generic_func, target_dtype: str, dialect_dtype: str):
    with FunctionBuilder(func_name, kind='cuda_device', ret_type=data_type(target_dtype)) as fb:
        # params
        x = Var('x', type=data_type(target_dtype))
        fb.extend_params([x])
        # body
        sb = StmtBuilder()
        sb += ReturnStmt(cast(generic_func(cast(x, dialect_dtype)), target_dtype))
        fb.set_body(sb.finish())
    register_primitive_function(name=func_name, func_or_type=fb.get())


def register_binary_dialect_primitive_function(func_name, generic_func, target_dtype: str, dialect_dtype: str):
    with FunctionBuilder(func_name, kind='cuda_device', ret_type=data_type(target_dtype)) as fb:
        # params
        x = Var('x', type=data_type(target_dtype))
        y = Var('y', type=data_type(target_dtype))
        fb.extend_params([x, y])
        # body
        sb = StmtBuilder()
        sb += ReturnStmt(cast(generic_func(cast(x, dialect_dtype), cast(y, dialect_dtype)), target_dtype))
        fb.set_body(sb.finish())
    register_primitive_function(name=func_name, func_or_type=fb.get())


def call_cuda(func_name, args: List[Expr]) -> Call:
    # todo: replace all usage of this function to call_primitive_func
    entry = primitive_func_pool.lookup_by_name('cuda_{}'.format(func_name))
    return Call(entry.var, args)
