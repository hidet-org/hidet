from typing import List
from hidet.graph.ir.functors import GraphRewriter
from hidet.graph.ir.flow_graph import FlowGraph, Operator, Tensor
from hidet.graph.ops import definitions as defs
from hidet.graph import ops
from hidet.ir.type import ScalarType
from .base import GraphPass
from hidet.utils import strict_zip, same_list


class DefaultTransformStrategy:
    always = [
        # defs.Conv2dOp, defs.MatmulOp, defs.AddOp, defs.SubOp, defs.MultiplyOp, defs.DivideOp, defs.PadOp
    ]
    never = [
        # defs.ErfOp, defs.PowOp,
        # defs.ReduceSumOp, defs.ReduceMeanOp  # disable because we can accumulate with higher precision
    ]


class AutoMixPrecisionRewriter(GraphRewriter):
    def __init__(self, target_dtype: str):
        super().__init__()
        assert ScalarType(target_dtype).is_float()
        self.target_dtype = target_dtype
        self.policy = DefaultTransformStrategy()

    @staticmethod
    def cast_float(x: Tensor, target_dtype: str) -> Tensor:
        if ScalarType(x.dtype).is_float():
            return ops.cast(x, target_dtype)
        return x

    def visit_Operator(self, op: Operator):
        recv_inputs: List[Tensor] = [self(v) for v in op.inputs]

        # if type(op) in self.policy.always:
        #     decision = 'always'
        if type(op) in self.policy.never:
            decision = 'never'
        else:
            decision = 'always'

        casted_inputs = []
        for orig_input, recv_input in strict_zip(op.inputs, recv_inputs):
            if decision == 'always':
                casted_inputs.append(self.cast_float(recv_input, self.target_dtype))
            elif decision == 'never':
                casted_inputs.append(self.cast_float(recv_input, orig_input.dtype))
            else:
                casted_inputs.append(recv_input)
        if same_list(casted_inputs, op.inputs):
            return op
        else:
            updated_outputs = op.reforward(casted_inputs)
            for original, updated in zip(op.outputs, updated_outputs):
                self.memo[original] = updated


class AutoMixPrecisionPass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        target_dtype = self.current_context().configs['precision']
        if target_dtype is None:
            return graph
        else:
            rewriter = AutoMixPrecisionRewriter(target_dtype)
            graph = rewriter(graph)
            return graph


def automatic_mix_precision_pass() -> GraphPass:
    return AutoMixPrecisionPass()
