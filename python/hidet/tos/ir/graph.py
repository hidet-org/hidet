from typing import List, Union, Dict, Set, Optional, Tuple
import warnings
import json
from collections import defaultdict
from hidet.tos.tensor import Tensor
from hidet.tos.operator import Operator


class FlowGraph:
    def __init__(self, outputs: List[Tensor], inputs=None, nodes=None):
        self.outputs: List[Tensor] = outputs
        self.inputs: Optional[List[Tensor]] = inputs
        self.nodes: Optional[List[Operator]] = nodes

    def __call__(self, *inputs: Tensor) -> List[Tensor]:
        return self.forward(*inputs)

    def forward(self, *inputs: Tensor) -> List[Tensor]:
        assert len(inputs) == len(self.inputs)
        assert all(input.storage is not None for input in inputs), 'Please feed non-symbolic tensor'
        tmap: Dict[Tensor, Tensor] = {}
        for st, at in zip(self.inputs, inputs):
            tmap[st] = at
        for node in self.nodes:
            node_inputs = [tmap[st] if st.storage is None else st for st in node.inputs]
            node_outputs = node.imperative_run(node_inputs)
            for st, at in zip(node.outputs, node_outputs):
                tmap[st] = at
        return [tmap[st] for st in self.outputs]

    def update_nodes(self):
        inputs, self.nodes = self._analyze(self.outputs)
        if self.inputs:
            if len(inputs) != len(self.inputs):
                raise ValueError('Found {} symbol inputs, but {} given'.format(len(inputs), len(self.inputs)))
            if any(a not in self.inputs for a in inputs):
                raise ValueError('There is a symbol tensor not given in inputs')
        else:
            if len(inputs) > 1:
                warnings.warn('There are {} symbol inputs traced, '
                              'but the inputs has not given to specify the order.'.format(len(inputs)))
            self.inputs = inputs
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
                if it.op is None:
                    continue
                v: Operator = it.op
                if v not in all_nodes:
                    find_all_nodes(v)
        for ot in outputs:
            if ot.trace:
                find_all_nodes(ot.op)

        # topological sort
        out_degree: Dict[Operator, int] = {u: 0 for u in all_nodes}
        for u in all_nodes:
            for it in u.inputs:
                if it.trace is None:
                    continue
                v = it.trace[0]
                out_degree[v] += 1
        for u in outputs:
            if u.op:
                out_degree[u.op] += 1

        stack: List[Operator] = []
        for u in outputs:
            if u.op:
                out_degree[u.op] -= 1
                if out_degree[u.op] == 0:
                    stack.append(u.op)
        while len(stack) > 0:
            op = stack.pop()
            nodes.append(op)
            for it in op.inputs:
                if it.op is None:
                    if it.storage is None and it not in inputs:
                        # input
                        inputs.append(it)
                else:
                    out_degree[it.op] -= 1
                    if out_degree[it.op] == 0:
                        stack.append(it.op)
        nodes = list(reversed(nodes))
        assert len(nodes) == len(all_nodes), 'all_nodes {} topo_order {}'.format(len(all_nodes), len(nodes))
        return inputs, nodes


def trace_from(tensor: Union[Tensor, List[Tensor]], inputs: Optional[Union[Tensor, List[Tensor]]] = None) -> FlowGraph:
    """
    Trace the flow graph given the output tensor(s).

    Parameters
    ----------
    tensor: Tensor or List[Tensor]
        The output tensor(s) that we trace from.
    inputs: Optional, Tensor or List[Tensor]
        The inputs of the flow graph. When there is only a single symbol tensor in the flow graph, it is
        optional. When there are multiple inputs, this is required to specify the input order.

    Returns
    -------
    ret: FlowGraph
        The flow graph that outputs the given input tensor(s).
    """
    if isinstance(tensor, Tensor):
        outputs = [tensor]
    else:
        outputs = list(tensor)
    if inputs is not None:
        if isinstance(inputs, Tensor):
            inputs = [inputs]
        else:
            inputs = list(inputs)
    return FlowGraph(outputs, inputs).update_nodes()
