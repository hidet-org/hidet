from typing import Sequence
from hidet.ir.dtypes import int32
from hidet.ir.expr import Call
from hidet.ir.primitives.cuda.sync import bar_sync
from hidet.ir.stmt import Stmt, EvaluateStmt
from hidet.ir.builders import StmtBuilder
from hidet.lang.constructs.context import HidetContext
from hidet.lang.cuda import threadIdx
from hidet.lang.cuda.contexts import WarpGroupContext

from hidet.ir.cute.expr import CallOp
from hidet.ir.primitives.cuda.setmaxnreg import setmaxnreg

from hidet.ir.functors import IRRewriter
from hidet.logging import logger


class CuteContextRewriter(IRRewriter):
    def __init__(self, context: HidetContext):
        super().__init__()
        self.context = context

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        if isinstance(stmt.expr, Call):
            func_var = stmt.expr.func_var
            func_name = func_var.name
            if func_name == "cuda_syncthreads":
                logger.warning(
                    "Function syncthreads cannot appear inside warpgroup context manager"
                    "and will be replaced with bar_sync, which will introduce unexpected"
                    "behavior. Please double-check the synchronization is expected."
                )
                return EvaluateStmt(bar_sync(self.context.num_threads))
        return super().visit_EvaluateStmt(stmt)

    def visit_CallOp(self, call: CallOp):
        op = self.visit(call.op)
        context_annotations = self.context.annotations
        return op.reforward(op.args, annotations_update=context_annotations).make_call()


class CuteWarpGroupsContext(WarpGroupContext):
    def __init__(self, group_ids: Sequence[int], num_regs: int, role: str):
        super().__init__(group_ids)
        self.num_regs = num_regs
        self.num_threads = len(set(group_ids)) * 128
        self.role = role
        self._annotations = {"group_ids": group_ids, "group_threads": self.num_threads, "role": role}

    @property
    def annotations(self):
        return self._annotations

    def post_process(self, stmt: Stmt):
        sb = StmtBuilder()
        rewriter = CuteContextRewriter(self)

        action = "dec" if self.role == "producer" else "inc"
        with sb.if_then(self.condition):
            sb.append(EvaluateStmt(setmaxnreg(self.num_regs, action)))
            sb.append(rewriter(stmt))

        return sb.finish()


def tid_in_groups(group_ids: Sequence[int]):
    min_warpgroup_id: int = min(group_ids)
    num_warpgroups: int = max(group_ids) - min(group_ids) + 1
    return (threadIdx.x - int32(min_warpgroup_id * 128)) % int32(num_warpgroups * 128)


def warp_groups_producer(group_ids: Sequence[int], num_regs: int):
    return CuteWarpGroupsContext(group_ids, num_regs, "producer")


def warp_groups_consumer(group_ids: Sequence[int], num_regs: int):
    return CuteWarpGroupsContext(group_ids, num_regs, "consumer")
