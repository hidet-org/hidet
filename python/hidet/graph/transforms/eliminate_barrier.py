from hidet.graph.ir import FlowGraph, Operator, Tensor, GraphRewriter
from hidet.graph.transforms import GraphPass

from .utils import is_barrier


class EliminateBarrierRewriter(GraphRewriter):
    def visit_Operator(self, op: Operator):
        inputs = [self(x) for x in op.inputs]

        if is_barrier(op):
            outputs = inputs
            for original, updated in zip(op.outputs, outputs):
                self.memo[original] = updated
        else:
            return GraphRewriter.visit_Operator(self, op)


class EliminateBarrierPass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        rewriter = EliminateBarrierRewriter()
        return rewriter(graph)


def eliminate_barrier_pass():
    return EliminateBarrierPass()
