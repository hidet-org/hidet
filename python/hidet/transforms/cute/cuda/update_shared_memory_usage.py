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
from hidet.ir.functors import IRVisitor, IRRewriter
from hidet.ir.expr import Constant, is_constant, if_then_else
from hidet.ir.cute.expr import CallOp

from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass
from hidet.ir.stmt import DeclareStmt, EvaluateStmt
from .lower_ops import request_smem_nbytes


class AnalyzeSharedMemoryUsage(IRVisitor):
    def __init__(self):
        super().__init__()
        self.dynamic_smem_bytes: int = 0

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            op = call.op
            self.dynamic_smem_bytes = max(request_smem_nbytes(op), self.dynamic_smem_bytes)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        if isinstance(stmt.expr, CallOp):
            call = stmt.expr
            op = call.op
            self.dynamic_smem_bytes = max(request_smem_nbytes(op), self.dynamic_smem_bytes)


class ApplySharedMemoryUsageUpdate(IRRewriter):
    def __init__(self, dynamic_smem_bytes: int):
        super().__init__()
        self.dynamic_smem_bytes = dynamic_smem_bytes

    def visit_Function(self, func: Function):
        func = super().visit_Function(func)
        if func.kind == "cuda_kernel":
            dynamic_smem_bytes = 0
            if "cuda.dynamic_smem_bytes" in func.attrs:
                dynamic_smem_bytes = func.attrs["cuda.dynamic_smem_bytes"]
            values = (dynamic_smem_bytes, self.dynamic_smem_bytes)
            if all(is_constant(val) for val in values):
                values = [v.value if isinstance(v, Constant) else v for v in values]
                dynamic_smem_bytes = max(values[0], values[1])
            else:
                dynamic_smem_bytes = if_then_else(values[0] > values[1], values[0], values[1])
            func.attrs["cuda.dynamic_smem_bytes"] = dynamic_smem_bytes
        return func


class UpdateSharedMemoryUsagePass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        analyze_shared_memory_usage = AnalyzeSharedMemoryUsage()
        analyze_shared_memory_usage.visit(func)

        rewriter = ApplySharedMemoryUsageUpdate(analyze_shared_memory_usage.dynamic_smem_bytes)
        return rewriter(func)


def update_shared_memory_usage_pass() -> FunctionPass:
    return UpdateSharedMemoryUsagePass()
