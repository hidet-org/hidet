from typing import List, Dict, Union, Tuple, Type, Optional
from collections import defaultdict

from hidet.tos.graph import FlowGraph, Operator, Tensor
from hidet.tos.transforms import GraphVisitor


class GraphUsageAnalyzer(GraphVisitor):
    def __init__(self):
        super().__init__()
        self.usage: Dict[Tensor, List[Tuple[Optional[Operator], int]]] = defaultdict(list)

    def analyze(self, graph: FlowGraph):
        self.usage = defaultdict(list)
        self.visit(graph)
        return self.usage

    def visit_FlowGraph(self, graph: FlowGraph):
        for idx, output in enumerate(graph.outputs):
            self(output)
            self.usage[output].append((None, idx))
        GraphVisitor.visit_FlowGraph(self, graph)

    def visit_Operator(self, op: Operator):
        for idx, input in enumerate(op.inputs):
            self.usage[input].append((op, idx))
        GraphVisitor.visit_Operator(self, op)


def collect(obj: Union[FlowGraph, Operator, Tensor], cls: Type[Union[Operator, Tensor]]):
    visitor = GraphVisitor()
    visitor.visit(obj)
    return [v for v in visitor.memo if isinstance(v, cls)]


def analyze_usage(graph: FlowGraph) -> Dict[Tensor, List[Tuple[Operator, int]]]:
    analyzer = GraphUsageAnalyzer()
    return analyzer.analyze(graph)
