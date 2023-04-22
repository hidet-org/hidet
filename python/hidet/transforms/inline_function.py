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
from typing import List, Dict
from hidet.ir.type import ReferenceType
from hidet.ir.expr import Call
from hidet.ir.func import IRModule, Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tools import collect
from hidet.ir.primitives import is_primitive_function, lookup_primitive_function
from hidet.ir.utils.call_graph import CallGraph, CallGraphNode
from hidet.transforms import Pass


class InlineFunctionRewriter(IRRewriter):
    pass


class InlineFunctionPass(Pass):

    def lets_inline(self, func: Function) -> bool:
        if func.kind in ['cuda_kernel', 'host_kernel', 'packed_func']:
            return False
        if any(isinstance(param.type, ReferenceType) for param in func.params):
            # TODO: support inline functions with reference parameters
            # like void func(int& a) {...}
            return False
        return True

    def process_module(self, ir_module: IRModule) -> IRModule:
        call_graph = CallGraph(ir_module)
        updated_ir_module = IRModule(task=ir_module.task)
        for node in call_graph.reversed_order:
            assert isinstance(node, CallGraphNode)
            if self.lets_inline(node.func):
                pass
            else:
                pass


def inline_function_pass():
    return InlineFunctionPass()
