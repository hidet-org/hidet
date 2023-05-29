from hidet.ir.builders import StmtBuilder
from hidet.ir.expr import logical_and, logical_or
from hidet.ir.functors import IRRewriter
from hidet.ir.primitives import printf
from hidet.ir.stmt import Stmt, LaunchKernelStmt, AssertStmt
from hidet.transforms.base import Pass, FunctionBodyPass


class CheckLaunchConfigurationRewriter(IRRewriter):
    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        sb = StmtBuilder()
        # if we launch a kernel with 0 dimension, cuda will complain with "cudaErrorInvalidConfiguration"
        # so we need to check the dimension > 0 before launching the kernel
        conditions = [dim > 0 for dim in stmt.grid_dim + stmt.block_dim]
        with sb.if_then(logical_and(*conditions)):
            upper_bounds = [
                2147483647,  # gridDim.x <= 2^31 - 1
                65535,  # gridDim.y <= 2^16 - 1
                65535,  # gridDim.z <= 2^16 - 1
                1024,  # blockDim.x <= 1024
                1024,  # blockDim.y <= 1024
                64,  # blockDim.z <= 64
            ]
            conditions = [dim > upper_bound for dim, upper_bound in zip(stmt.grid_dim + stmt.block_dim, upper_bounds)]
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
            sb += stmt
        return sb.finish()


class CheckLaunchConfigurationPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return CheckLaunchConfigurationRewriter().rewrite(stmt)


def check_launch_configuration_pass() -> Pass:
    return CheckLaunchConfigurationPass()
