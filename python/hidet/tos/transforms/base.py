from typing import Union
from hidet.utils.py import same_list
from hidet.tos.graph import FlowGraph, Operator, Tensor
from hidet.utils import Timer


class GraphVisitor:
    def __init__(self):
        self.memo = {}

    def __call__(self, obj):
        return self.visit(obj)

    def visit(self, obj: Union[FlowGraph, Operator, Tensor, list, tuple]):
        key = obj if not isinstance(obj, list) else id(obj)
        if self.memo is not None and obj in self.memo:
            return self.memo[key]
        if isinstance(obj, FlowGraph):
            self.visit_FlowGraph(obj)
        elif isinstance(obj, Operator):
            self.visit_Operator(obj)
        elif isinstance(obj, Tensor):
            self.visit_Tensor(obj)
        elif isinstance(obj, (list, tuple)):
            self.visit_Sequence(obj)
        else:
            raise ValueError(type(obj))
        if self.memo is not None:
            self.memo[key] = None

    def visit_FlowGraph(self, graph: FlowGraph):
        for output in graph.outputs:
            self(output)

    def visit_Operator(self, op: Operator):
        for input in op.inputs:
            self(input)

    def visit_Tensor(self, tensor: Tensor):
        if tensor.trace is None:
            return tensor
        self(tensor.trace[0])

    def visit_Sequence(self, seq: Union[list, tuple]):
        for obj in seq:
            self(obj)


class GraphRewriter:
    def __init__(self):
        self.memo = {}

    def __call__(self, obj):
        return self.visit(obj)

    def visit(self, obj: Union[FlowGraph, Operator, Tensor, list, tuple]):
        key = obj if not isinstance(obj, list) else id(obj)
        if self.memo and obj in self.memo:
            return self.memo[key]
        if isinstance(obj, FlowGraph):
            ret = self.visit_FlowGraph(obj)
        elif isinstance(obj, Operator):
            ret = self.visit_Operator(obj)
        elif isinstance(obj, Tensor):
            ret = self.visit_Tensor(obj)
        elif isinstance(obj, (list, tuple)):
            ret = self.visit_Sequence(obj)
        else:
            raise ValueError(type(obj))
        if self.memo:
            self.memo[key] = ret
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
            new_op = op.__class__(*inputs, **op.attributes)
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

    def visit_Sequence(self, seq: Union[list, tuple]):
        return seq.__class__([self(obj) for obj in seq])


class GraphPass:
    def __call__(self, graph: FlowGraph) -> FlowGraph:
        with Timer('{:>30}'.format(self.__class__.__name__), verbose=True):
            return self.process_graph(graph)

    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        raise NotImplementedError()


