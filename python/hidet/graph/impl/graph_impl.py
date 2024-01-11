from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from hidet.graph.tensor import Tensor
from hidet.graph.operator import Operator
from hidet.graph.flow_graph import FlowGraph


def graph_analyze(
    outputs: List[Tensor],
    stop_tensors: Optional[List[Tensor]] = None
) -> Tuple[List[Tensor], List[Operator], Dict[Tensor, int]]:
    """
    Analyze the implicit flow graph by backwards traversing the graph from given outputs.

    Parameters
    ----------
    outputs: List[Tensor]
        The outputs of the flow graph to traversing from.

    stop_tensors: List[Tensor], optional
        The tensors that we should stop traversing when we reach them, even if they have non-None trace attribute.
        When stop_tensors is None, we will stop traversing when we reach the tensors that have None trace attribute.

    Returns
    -------
    free_vars, nodes, usage_count: Tuple[List[Tensor], List[Operator], Dict[Tensor, int]]
        The free variables, nodes and usage count of the flow graph.

        The free variables are the free symbolic tensors that are not produced by any operators and do not contain
        the non-None storage attribute, nor the stop_tensors.

        The nodes are the operators that are used to produce the outputs, in topological order.

        The usage count contains the number of times each tensor is used.
    """
    free_vars = []
    nodes: List[Operator] = []
    stop_tensors: List[Tensor] = stop_tensors or []

    # find out all nodes
    all_nodes: Set[Operator] = set()

    def find_all_nodes(u: Operator):
        all_nodes.add(u)
        for x in u.inputs:
            if x.op is None or x in stop_tensors:
                continue
            v: Operator = x.op
            if v not in all_nodes:
                find_all_nodes(v)

    def valid(t: Tensor) -> bool:
        return t.op is not None and t.op not in all_nodes and t not in stop_tensors

    for ot in outputs:
        if ot.trace and ot not in stop_tensors:
            find_all_nodes(ot.op)

    # topological sort
    out_degree: Dict[Operator, int] = {u: 0 for u in all_nodes}
    for u in all_nodes:
        for it in filter(valid, u.inputs):
            out_degree[it.op] += 1
    for u in filter(valid, outputs):
        out_degree[u.op] += 1

    stack: List[Operator] = []
    for u in filter(valid, outputs):
        out_degree[u.op] -= 1
        if out_degree[u.op] == 0:
            stack.append(u.op)
    while len(stack) > 0:
        op = stack.pop()
        nodes.append(op)
        for it in op.inputs:
            if it.op is None:
                if it.storage is None and all(it is not v for v in free_vars) and it not in stop_tensors:
                    # a free variable
                    free_vars.append(it)
            elif it.op not in all_nodes:
                pass
            else:
                if it is not it.op.outputs[it.trace[1]]:
                    raise ValueError('The trace is broken')
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

    return free_vars, nodes, usage_count


def analyze_share_map(graph: FlowGraph) -> Dict[int, int]:
    pass

