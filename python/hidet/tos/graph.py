from typing import List, Union, Dict, Set
import json
from collections import defaultdict
from hidet.tos.tensor import Tensor
from hidet.tos.operator import Operator


class FlowGraph:
    def __init__(self):
        self.inputs: List[Tensor] = []
        self.nodes: List[Operator] = []
        self.outputs: List[Tensor] = []

    def __call__(self, *inputs: List[Tensor]) -> List[Tensor]:
        return self.forward(*inputs)

    def forward(self, *inputs: List[Tensor]) -> List[Tensor]:
        pass


def trace_from(tensor: Union[Tensor, List[Tensor]]) -> FlowGraph:
    graph = FlowGraph()
    if isinstance(tensor, Tensor):
        graph.outputs = [tensor]
    else:
        graph.outputs = list(tensor)
    # find out all nodes
    all_nodes: Set[Operator] = set()

    def find_all_nodes(u: Operator):
        all_nodes.add(u)
        for it in u.inputs:
            if it.trace is None:
                continue
            v: Operator = it.trace[0]
            if v not in all_nodes:
                find_all_nodes(v)
    for ot in graph.outputs:
        find_all_nodes(ot.trace[0])

    # topological sort
    out_degree: Dict[Operator: int] = {u: 0 for u in all_nodes}
    for u in all_nodes:
        for it in u.inputs:
            if it.trace is None:
                continue
            v = it.trace[0]
            out_degree[v] += 1

    nodes: List[Operator] = []

    def topo_sort(u: Operator):
        nodes.append(u)
        for it in u.inputs:
            if it.trace is None:
                if it not in graph.inputs and it.storage is None:
                    graph.inputs.append(it)
                continue
            v: Operator = it.trace[0]
            out_degree[v] -= 1
            if out_degree[v] == 0:
                topo_sort(v)
    for ot in graph.outputs:
        u = ot.trace[0]
        if u not in nodes:
            topo_sort(ot.trace[0])
    graph.nodes.extend(reversed(nodes))
    return graph

