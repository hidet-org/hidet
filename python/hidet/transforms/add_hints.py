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

import hidet

from hidet.ir.func import Function
from hidet.transforms import Pass
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import SeqStmt

from hidet.ir.primitives.debug import __builtin_assume


class AddHintsRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.hint_body = None

    def get_hints_body(self, func: Function):
        from hidet.lang.cuda import threadIdx, blockIdx
        from hidet.ir.analyzers import normalize_launch_dims

        block_dims = normalize_launch_dims(func.attrs['cuda.block_dim'])
        grid_dims = normalize_launch_dims(func.attrs['cuda.grid_dim'])

        with hidet.script_module() as script_module:

            @hidet.script
            def _hint_func():
                __builtin_assume(0 <= threadIdx.x)
                __builtin_assume(threadIdx.x < block_dims[0])
                __builtin_assume(0 <= threadIdx.y)
                __builtin_assume(threadIdx.y < block_dims[1])
                __builtin_assume(0 <= threadIdx.z)
                __builtin_assume(threadIdx.z < block_dims[2])

                __builtin_assume(0 <= blockIdx.x)
                __builtin_assume(blockIdx.x < grid_dims[0])
                __builtin_assume(0 <= blockIdx.y)
                __builtin_assume(blockIdx.y < grid_dims[1])
                __builtin_assume(0 <= blockIdx.z)
                __builtin_assume(blockIdx.z < grid_dims[2])

        hint = script_module.functions[0].body
        self.hint_body = hint
        return hint

    def visit_Function(self, func: Function):
        if func.kind != 'cuda_kernel':
            return func
        hint_body = self.get_hints_body(func)
        body = self.visit(func.body)
        body = SeqStmt((hint_body, body))
        return Function(func.name, func.params, body, func.ret_type, func.kind, func.attrs)


class AddHintsPass(Pass):
    def process_func(self, func: Function) -> Function:
        rewriter = AddHintsRewriter()
        res = rewriter.visit(func)
        return res


def add_hints_pass() -> Pass:
    return AddHintsPass()
