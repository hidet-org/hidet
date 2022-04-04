from hidet.tos.ir import FlowGraph, Operator, Tensor, GraphRewriter
from hidet.tos.transforms import GraphPass
from hidet import utils


class FoldConstantRewriter(GraphRewriter):
    @utils.line_profile()
    def visit_Operator(self, op: Operator):
        inputs = [self(input) for input in op.inputs]
        if all(input.storage is not None for input in inputs):
            outputs = Operator.imperative_run(op, inputs)
            for original, updated in zip(op.outputs, outputs):
                self.memo[original] = updated
            return None
        else:
            if utils.same_list(inputs, op.inputs):
                return op
            else:
                # new_op = op.__class__(*inputs, **op.attributes)
                new_op = op.clone(*inputs)
                for original, updated in zip(op.outputs, new_op.run()):
                    self.memo[original] = updated
                return new_op


class FoldConstantPass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        rewriter = FoldConstantRewriter()
        return rewriter(graph)


def fold_const_pass():
    return FoldConstantPass()
