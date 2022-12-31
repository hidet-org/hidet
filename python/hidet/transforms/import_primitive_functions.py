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
from hidet.ir.expr import Call
from hidet.ir.func import IRModule, Function
from hidet.ir.functors import collect
from hidet.ir.primitives import is_primitive_function, lookup_primitive_function
from hidet.transforms import Pass


class ImportPrimitiveFunctionPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        used_primitive_funcs = set()
        for func in ir_module.functions.values():
            calls: List[Call] = collect(func.body, Call)
            for call in calls:
                callee_name: str = call.func_var.hint
                if is_primitive_function(callee_name):
                    used_primitive_funcs.add(callee_name)

        primitive_funcs: List[Function] = []
        for func_name in used_primitive_funcs:
            entry = lookup_primitive_function(func_name)
            if entry.function is not None:
                primitive_funcs.append(entry.function)

        if len(primitive_funcs) == 0:
            return ir_module
        else:
            new_ir_module = IRModule(task=ir_module.task)
            for func_name, func in ir_module.functions.items():
                new_ir_module.add(func_name, func)
            for func in primitive_funcs:
                if func.name not in new_ir_module.functions:
                    new_ir_module.add(func.name, func)
            return new_ir_module


def import_primitive_functions_pass() -> Pass:
    return ImportPrimitiveFunctionPass()
