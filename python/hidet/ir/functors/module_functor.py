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
# pylint: disable=bad-staticmethod-argument
from hidet.ir.func import IRModule, Function
from hidet.utils import same_list

from .base_functor import BaseFunctor, BaseVisitor, BaseRewriter


class ModuleFunctor(BaseFunctor):
    def visit_dispatch(self, node):
        if isinstance(node, IRModule):
            return self.visit_IRModule(node)
        elif isinstance(node, Function):
            return self.visit_Function(node)
        else:
            return NotImplemented

    def visit_IRModule(self, module: IRModule):
        raise NotImplementedError()

    def visit_Function(self, func: Function):
        raise NotImplementedError()


class ModuleVisitor(ModuleFunctor, BaseVisitor):
    def visit_IRModule(self, module: IRModule):
        self.visit(module.global_vars)
        self.visit(module.functions)

    def visit_Function(self, func: Function):
        self.visit(func.params)
        self.visit(func.extern_vars)
        self.visit(func.ret_type)
        self.visit(func.body)
        self.visit(func.attrs)


class ModuleRewriter(ModuleFunctor, BaseRewriter):
    def visit_IRModule(self, module: IRModule):
        global_vars = self.visit(module.global_vars)
        functions = self.visit(module.functions)
        if same_list(global_vars, module.global_vars) and functions is module.functions:
            return module
        else:
            return IRModule(functions, module.task, global_vars)

    def visit_Function(self, func: Function):
        params = self.visit(func.params)
        ret_type = self.visit(func.ret_type)
        body = self.visit(func.body)
        attrs = self.visit(func.attrs)
        if same_list(params, func.params) and ret_type is func.ret_type and body is func.body and attrs is func.attrs:
            return func
        else:
            return Function(func.name, params, body, ret_type, func.kind, func.extern_vars, attrs)
