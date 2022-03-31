from typing import Union
from hidet.utils.py import same_list
from hidet.tos.graph import FlowGraph, Operator, Tensor


class GraphRewriter:
    def __init__(self):
        self.memo = {}

    def __call__(self, obj):
        return self.visit(obj)

    def visit(self, obj: Union[FlowGraph, Operator, Tensor]):
        if self.memo and obj in self.memo:
            return self.memo[obj]
        if isinstance(obj, FlowGraph):
            ret = self.visit_FlowGraph(obj)
        elif isinstance(obj, Operator):
            ret = self.visit_Operator(obj)
        elif isinstance(obj, Tensor):
            ret = self.visit_Tensor(obj)
        else:
            raise ValueError(type(obj))
        if self.memo:
            self.memo[obj] = ret
        return ret

    def visit_FlowGraph(self, graph: FlowGraph):
        outputs = [self.visit(output) for output in graph.outputs]
        if same_list(outputs, graph.outputs):
            return graph
        else:
            return FlowGraph(outputs)

    def visit_Operator(self, op: Operator):
        inputs = [self(input) for input in op.inputs]
        if same_list(inputs, op.inputs):
            return op
        else:
            new_op = Operator(inputs, op.task)
            for original, updated in zip(op.outputs, new_op.run()):
                self.memo[original] = updated
            return new_op

    def visit_Tensor(self, tensor: Tensor):
        if tensor.trace is None:
            # input
            return tensor
        self(tensor.trace[0])
        if tensor in self.memo:
            # the operator has been updated
            return self.memo[tensor]
        else:
            return tensor


class GraphPass:
    def __call__(self, graph: FlowGraph) -> FlowGraph:
        return self.process_graph(graph)

    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        raise NotImplementedError()


