from typing import List

from hidet.graph.flow_graph import FlowGraph
from hidet.graph.transforms import GraphPass
from .graph_patterns import SubgraphRewriteRule
from .subgraph_rewrite import SubgraphRewritePass


class SelectiveQuantizePass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        rewrite_patterns: List[SubgraphRewriteRule] = self.current_context().configs['quantize_patterns']
        graph = SubgraphRewritePass(rewrite_patterns)(graph)
        return graph


def selective_quantize_pass() -> GraphPass:
    return SelectiveQuantizePass()
