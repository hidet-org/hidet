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

from hidet.ir.expr import Var, SymbolVar, symbol_var
from hidet.ir.module import IRModule
from hidet.ir.functors import IRRewriter
from hidet.ir.primitives.vars import registered_primitive_variables, lookup_primitive_variable
from hidet.ir.type import DataType, data_type
from hidet.transforms import Pass


class UnifyGlobalObjectsRewriter(IRRewriter):
    def visit_DataType(self, t: DataType):
        return data_type(t.name)

    def visit_Var(self, var: Var):
        if isinstance(var, SymbolVar):
            assert isinstance(var.type, DataType)
            return symbol_var(var.name, var.type.name)
        elif var.name is not None and var.name in registered_primitive_variables:
            return lookup_primitive_variable(var.name)
        else:
            return super().visit_Var(var)


class UnifyGlobalObjectsPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        rewriter = UnifyGlobalObjectsRewriter()
        return rewriter.visit(ir_module)


def unify_global_objects_pass() -> Pass:
    return UnifyGlobalObjectsPass()
