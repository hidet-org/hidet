from typing import List, Union, Dict, Set, Optional, Tuple
import json
from collections import defaultdict
from hidet.tos.tensor import Tensor
from hidet.tos.operator import Operator


class FlowGraph:
    def __init__(self, outputs: List[Tensor], inputs=None, nodes=None):
        self.outputs: List[Tensor] = outputs
        self.inputs: Optional[List[Tensor]] = inputs
        self.nodes: Optional[List[Operator]] = nodes

    def __call__(self, *inputs: List[Tensor]) -> List[Tensor]:
        return self.forward(*inputs)

    def forward(self, *inputs: List[Tensor]) -> List[Tensor]:
        pass

    def finish(self):
        self.inputs, self.nodes = self._analyze(self.outputs)
        return self

    @staticmethod
    def _analyze(outputs: List[Tensor]) -> Tuple[List[Tensor], List[Operator]]:
        inputs = []
        nodes: List[Operator] = []
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
        for ot in outputs:
            find_all_nodes(ot.trace[0])

        # topological sort
        out_degree: Dict[Operator: int] = {u: 0 for u in all_nodes}
        for u in all_nodes:
            for it in u.inputs:
                if it.trace is None:
                    continue
                v = it.trace[0]
                out_degree[v] += 1

        def topo_sort(u: Operator):
            nodes.append(u)
            for it in u.inputs:
                if it.trace is None:
                    if it not in inputs and it.storage is None:
                        inputs.append(it)
                    continue
                v: Operator = it.trace[0]
                out_degree[v] -= 1
                if out_degree[v] == 0:
                    topo_sort(v)
        for ot in outputs:
            u = ot.trace[0]
            if u not in nodes:
                topo_sort(ot.trace[0])
        nodes = list(reversed(nodes))
        return inputs, nodes


def trace_from(tensor: Union[Tensor, List[Tensor]]) -> FlowGraph:
    if isinstance(tensor, Tensor):
        outputs = [tensor]
    else:
        outputs = list(tensor)
    return FlowGraph(outputs).finish()

