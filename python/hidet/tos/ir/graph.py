from __future__ import annotations
from typing import List, Union, Dict, Set, Optional, Tuple
import os
import pickle
import warnings
from collections import defaultdict
from hidet.tos.tensor import Tensor
from hidet.tos.operator import Operator
from hidet.utils import tracer


class FlowGraph:
    def __init__(self, outputs: List[Tensor], inputs=None, nodes=None):
        self.outputs: List[Tensor] = outputs
        self.inputs: Optional[List[Tensor]] = inputs
        self.nodes: Optional[List[Operator]] = nodes
        self.usage_count: Optional[Dict[Tensor, int]] = None

    def __call__(self, *inputs: Tensor) -> Union[List[Tensor], Tensor]:
        return self.forward(*inputs)

    def forward(self, *inputs: Tensor) -> Union[List[Tensor], Tensor]:
        if any(v is None for v in [self.inputs, self.nodes, self.usage_count]):
            self.update_nodes()
        if len(inputs) != len(self.inputs):
            raise ValueError('FlowGraph expects {} inputs, but got {}.'.format(len(self.inputs), len(inputs)))
        for idx, tensor in enumerate(inputs):
            if tensor.storage is None:
                raise ValueError('FlowGraph expects all input tensors are non-symbolic, '
                                 'but the input {} ({}) is a symbol tensor.'.format(idx, tensor.signature()))
        usage_count = self.usage_count.copy()
        tensor_map: Dict[Tensor, Tensor] = {}
        for st, at in zip(self.inputs, inputs):
            tensor_map[st] = at
        for node in self.nodes:
            # prepare node inputs
            node_inputs = []
            for node_input in node.inputs:
                if node_input.storage is None:
                    # symbolic input
                    node_inputs.append(tensor_map[node_input])
                    usage_count[node_input] -= 1
                    if usage_count[node_input] == 0:
                        # free the memory
                        del tensor_map[node_input]
                else:
                    # constant input
                    node_inputs.append(node_input)
            # run node
            args = {f'input_{idx}': f'{tensor.dtype}{tensor.shape}' for idx, tensor in enumerate(node.inputs)}
            args.update(node.attributes)
            with tracer.profile(node.name, category='op', args=args, trace_cuda=True):
                node_outputs = node.imperative_run(node_inputs)
            for st, at in zip(node.outputs, node_outputs):
                tensor_map[st] = at
        ret = [tensor_map[st] for st in self.outputs]
        return ret[0] if len(ret) == 1 else ret

    def save(self, fname: str):
        # before save, clear the packed func cache because ctypes object can not be pickled
        for node in self.nodes:
            node.task_func = None
        self.usage_count, self.inputs, self.nodes = None, None, None

        dirname = os.path.dirname(fname)
        os.makedirs(dirname, exist_ok=True)
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname: str) -> FlowGraph:
        with open(fname, 'rb') as f:
            ret = pickle.load(f)
        if not isinstance(ret, FlowGraph):
            raise TypeError('Expect to load FlowGraph, got {}'.format(type(ret)))
        ret.update_nodes()
        return ret

    def update_nodes(self):
        inputs, self.nodes, self.usage_count = self._analyze(self.outputs)
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
    def _analyze(outputs: List[Tensor]) -> Tuple[List[Tensor], List[Operator], Dict[Tensor, int]]:
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
                if it.op is None:
                    continue
                out_degree[it.op] += 1
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

        # tensor usage count
        usage_count: Dict[Tensor, int] = defaultdict(int)
        for op in all_nodes:
            for inp in op.inputs:
                usage_count[inp] += 1
        for graph_output in outputs:
            usage_count[graph_output] += 1

        return inputs, nodes, usage_count


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
        if tensor.trace is None:
            raise ValueError('trace_from expects symbol tensor(s).')
        outputs = [tensor]
    else:
        outputs = list(tensor)
    if inputs is not None:
        if isinstance(inputs, Tensor):
            inputs = [inputs]
        else:
            inputs = list(inputs)
    return FlowGraph(outputs, inputs).update_nodes()


def save_graph(graph: FlowGraph, fname: str):
    graph.save(fname)


def load_graph(fname: str) -> FlowGraph:
    return FlowGraph.load(fname)
