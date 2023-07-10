from typing import List, Optional, Dict, Tuple, Set
import logging

from hidet.graph.flow_graph import FlowGraph, Operator, Tensor
from hidet.graph.transforms import GraphPass, PassContext
from hidet.graph.graph_utils.functors import analyze_usage, graph_collect
from hidet.graph import graph_utils
from hidet.utils import strict_zip
from .graph_patterns import SubgraphRewriteRule, TensorPattern, OperatorPattern, MatchDict, Usage, graph_pattern_match
from .graph_patterns.base import registered_rewrite_rules, register_rewrite_rule
from .subgraph_rewrite import SubgraphRewritePass


class SelectiveQuantizePass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        rewrite_patterns: List[SubgraphRewriteRule] = self.current_context().configs['quantize_patterns']
        graph = SubgraphRewritePass(rewrite_patterns)(graph)
        return graph
    

def selective_quantize_pass() -> GraphPass:
    return SelectiveQuantizePass()
