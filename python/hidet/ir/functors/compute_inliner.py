from .base import ExprRewriter, same_list
from .util_functors import collect, rewrite
from hidet.ir.dialects.compute import TensorInput, TensorCompute, ScalarInput, ReduceCompute, ComputeNode
from hidet.ir.expr import Var, TensorElement


class ComputeInlineRewriter(ExprRewriter):
    def visit_TensorElement(self, e: TensorElement):
        base = self(e.base)
        indices = [self(idx) for idx in e.indices]
        if isinstance(base, TensorCompute):
            return rewrite(base.value, {axis: index for axis, index in zip(base.axes, indices)})
        else:
            return e


def inline_compute(expr) -> ComputeNode:
    inliner = ComputeInlineRewriter()
    return inliner(expr)
