from .base import ExprRewriter
from .util_functors import collect, rewrite
from hidet.ir.dialects.compute import TensorInput, TensorCompute, ScalarInput, ReduceCompute
from hidet.ir.expr import Var, TensorElement


class ComputeInlineRewriter(ExprRewriter):
    def visit_TensorElement(self, e: TensorElement):
        if isinstance(e.base, TensorCompute):
            exprs = collect(e.base.value, node_types=(TensorCompute, ReduceCompute))
            if len(exprs) == 0:
                return rewrite(e.base.value, {axis: index for axis, index in zip(e.base.axes, e.indices)})
        return e


def inline_compute(expr):
    inliner = ComputeInlineRewriter()
    return inliner(expr)
