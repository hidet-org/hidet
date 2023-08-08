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
from hidet.ir.builders import StmtBuilder
from hidet.ir.expr import logical_and, logical_or
from hidet.ir.stmt import BlackBoxStmt
from hidet.ir.functors import IRRewriter
from hidet.ir.primitives import printf
from hidet.ir.primitives.cuda import check_cuda_error
from hidet.ir.stmt import LaunchKernelStmt, AssertStmt
from hidet.ir.func import Function
from hidet.transforms.base import Pass, FunctionPass


class CheckLaunchConfigurationRewriter(IRRewriter):
    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        sb = StmtBuilder()
        # if we launch a kernel with 0 dimension, cuda will complain with "cudaErrorInvalidConfiguration"
        # so we need to check the dimension > 0 before launching the kernel
        conditions = [dim > 0 for dim in stmt.grid_dim + stmt.block_dim]
        with sb.if_then(logical_and(*conditions)):
            upper_bounds = [
                # 2147483647,  # gridDim.x <= 2^31 - 1, we don't need to check this because it's unlikely to reach
                65535,  # gridDim.y <= 2^16 - 1
                65535,  # gridDim.z <= 2^16 - 1
                1024,  # blockDim.x <= 1024
                1024,  # blockDim.y <= 1024
                64,  # blockDim.z <= 64
            ]
            conditions = [
                dim > upper_bound for dim, upper_bound in zip(stmt.grid_dim[1:] + stmt.block_dim, upper_bounds)
            ]
            with sb.if_then(logical_or(*conditions)):
                sb += printf(
                    "Launching kernel with grid_dim = (%d, %d, %d), block_dim = (%d, %d, %d)\n",
                    stmt.grid_dim[0],
                    stmt.grid_dim[1],
                    stmt.grid_dim[2],
                    stmt.block_dim[0],
                    stmt.block_dim[1],
                    stmt.block_dim[2],
                )
                sb += AssertStmt(False, "Invalid launch configuration")
            with sb.if_then(stmt.shared_mem_bytes > 49152):
                # if the shared memory is larger than 48KB, we should call cudaFuncSetAttribute
                sb += BlackBoxStmt(
                    "cudaFuncSetAttribute({}, cudaFuncAttributeMaxDynamicSharedMemorySize, {});",
                    stmt.func_var,
                    stmt.shared_mem_bytes,
                )
                sb += check_cuda_error()
            sb += stmt
            sb += check_cuda_error()
        return sb.finish()


class CheckLaunchConfigurationPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        return CheckLaunchConfigurationRewriter().rewrite(func)


def check_launch_configuration_pass() -> Pass:
    return CheckLaunchConfigurationPass()
