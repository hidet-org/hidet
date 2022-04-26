from typing import List, Union
from .base import ExprRewriter, ExprVisitor, same_list, Node
from .util_functors import collect, rewrite
from hidet.ir.dialects.compute import TensorInput, TensorCompute, ScalarInput, ReduceCompute, ComputeNode
from hidet.ir.expr import Var, TensorElement, Expr
from hidet.utils import prod


class ComponentCollector(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.inputs: List[TensorInput] = []
        self.computes: List[TensorCompute] = []
        self.reduces: List[ReduceCompute] = []

    def collect(self, e: TensorCompute):
        self.inputs.clear()
        self.computes.clear()
        self.reduces.clear()
        ExprVisitor.visit_TensorCompute(self, e)
        return self.inputs, self.computes, self.reduces

    def visit_TensorInput(self, e: TensorInput):
        self.inputs.append(e)

    def visit_TensorCompute(self, e: TensorCompute):
        self.computes.append(e)

    def visit_ReduceCompute(self, e: ReduceCompute):
        self.reduces.append(e)


class ComputeInlineRewriter(ExprRewriter):
    def __init__(self, reduce_limit=0):
        super().__init__()
        self.reduce_limit = reduce_limit

    def visit_TensorElement(self, e: TensorElement):
        base = self(e.base)
        if isinstance(base, TensorCompute):
            inputs, computes, reduces = ComponentCollector().collect(base)
            cnt = sum(prod(reduce.const_shape()) for reduce in reduces)
            if cnt <= self.reduce_limit:
                return rewrite(base.value, {axis: index for axis, index in zip(base.axes, e.indices)})
        return e


def inline_compute(expr: TensorCompute, reduce_limit=0) -> TensorCompute:
    inliner = ComputeInlineRewriter(reduce_limit)
    return inliner(expr)
