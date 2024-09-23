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

import logging
from hidet.ir.module import IRModule
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LaunchKernelStmt
from hidet.transforms import Pass


class AttachHashToSignatureRewriter(IRRewriter):
    def __init__(self, old_name: str, new_name: str, use_memo=True):
        self.old_name = old_name
        self.new_name = new_name
        super().__init__(use_memo)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        if stmt.func_var.name == self.old_name:
            stmt.func_var.name = self.new_name
        return super().visit_LaunchKernelStmt(stmt)


class AttachHashToSignature(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        if ir_module.task is None:
            logging.warning("A IRModule without task is detected. Designated function hash cannot be append")
            return ir_module

        modify_name = {}

        for func in ir_module.functions.values():
            if func.kind in ['cuda_kernel', 'cpu_kernel']:
                task_hash = ir_module.task.calculate_hash(4)
                old_name = func.name
                new_name = func.name + f'_{task_hash}'
                modify_name[old_name] = new_name

        if not modify_name:
            return ir_module

        new_ir_module = ir_module.copy()

        for old_name, new_name in modify_name.items():
            new_ir_module.functions[new_name] = new_ir_module.functions.pop(old_name)
            new_ir_module.functions[new_name].name = new_name

            if old_name in ir_module.global_vars.keys():
                new_ir_module.global_vars[new_name] = new_ir_module.global_vars.pop(old_name)
                new_ir_module.global_vars[new_name].name = new_name

        if any(func.name.startswith('launch') for func in new_ir_module.functions.values() if func.kind == 'public'):
            for old_name, new_name in modify_name.items():
                rewriter = AttachHashToSignatureRewriter(old_name, new_name)
                new_ir_module = rewriter.visit(new_ir_module)

        return new_ir_module


def attach_hash_to_signature() -> Pass:
    return AttachHashToSignature()
