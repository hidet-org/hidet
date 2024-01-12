from typing import List, Tuple, Dict, Set, Optional, Union
from collections import defaultdict
import hidet.option
from hidet.graph.tensor import Tensor
from hidet.graph.operator import Operator
from hidet.graph.flow_graph import FlowGraph
from hidet.utils.doc import Doc, NewLine, Text, doc_join


def graph_analyze(
    outputs: List[Tensor], stop_tensors: Optional[List[Tensor]] = None
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
        return t.op is not None and t.op in all_nodes and t not in stop_tensors

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


def graph_analyze_share_map(graph: FlowGraph) -> Dict[int, int]:
    """
    Analyze to get the share map of the flow graph.

    Hidet allows the output tensors of a graph to share the memory with some input tensors of the graph. This function
    analyzes the share map that stores the mapping from the output index to the index of the input tensor that the
    output tensor shares the memory with.

    Parameters
    ----------
    graph: FlowGraph
        The flow graph to analyze.

    Returns
    -------
    share_map: Dict[int, int]
        The share map of the flow graph.
    """
    share_map: Dict[int, int] = {}
    tensor_share_map: Dict[Tensor, Tensor] = {}
    for node in graph.nodes:
        for output_index, output in enumerate(node.outputs):
            if node.share_map and output_index in node.share_map:
                input_index = node.share_map[output_index]
                tensor_share_map[output] = tensor_share_map.get(node.inputs[input_index], node.inputs[input_index])
    for output_index, output in enumerate(graph.outputs):
        if output in tensor_share_map:
            shared_tensor = tensor_share_map[output]
            if shared_tensor in graph.inputs:
                input_index = graph.inputs.index(shared_tensor)
                share_map[output_index] = input_index
            else:
                raise ValueError(
                    'The following output tensor of the graph shares the memory with a graph intermediate tensor:\n'
                    '  Output {}: {}'.format(output_index, output.signature())
                )
    return share_map


def graph_as_text(graph: FlowGraph) -> str:
    """
    Get a human-readable text representation of the flow graph.

    Parameters
    ----------
    graph: FlowGraph
        The flow graph to be represented.

    Returns
    -------
    ret: str
        The human-readable text representation of the flow graph.
    """
    from hidet.ir.tools.printer import IRPrinter
    from hidet.ir.expr import Expr
    from hidet.graph.operator import SymbolVar

    printer = IRPrinter()
    size_var_equivalence: Dict[SymbolVar, SymbolVar] = {}

    def get_tensor_sig(x: Tensor) -> Doc:
        shape_items = []
        for i in range(len(x.shape)):
            if x.trace is None:
                shape_items.append(printer(x.shape[i]))
            else:
                op: Operator
                op, idx = x.trace
                task_out_dim = op.task.outputs[idx].shape[i]
                if isinstance(task_out_dim, SymbolVar) and task_out_dim in size_var_equivalence:
                    shape_items.append(printer(size_var_equivalence[task_out_dim]))
                else:
                    shape_items.append(printer(x.shape[i]))

        return Text(x.dtype.name) + '[' + doc_join(shape_items, ', ') + '][' + Text(x.device.kind) + ']'

    def get_attr_repr(value: Union[float, int, bool, str, list, tuple, FlowGraph]) -> Doc:
        if isinstance(value, (float, int, bool)):
            return Text(str(value))
        elif isinstance(value, str):
            return Text('"{}"'.format(value))
        elif isinstance(value, list):
            return '[' + doc_join([get_attr_repr(v) for v in value], ', ') + ']'
        elif isinstance(value, tuple):
            return '(' + doc_join([get_attr_repr(v) for v in value], ', ') + ')'
        elif isinstance(value, FlowGraph):
            return Text('FlowGraph({})'.format(', '.join(u.name for u in value.nodes)))
        elif isinstance(value, Expr):
            return printer(value)
        else:
            return Text(str(value))

    def get_comment(op: Operator) -> Doc:
        items = []
        for out, task_output in zip(op.outputs, op.task.outputs):
            for dim, task_dim in zip(out.shape, task_output.shape):
                if isinstance(dim, SymbolVar) and task_dim not in size_var_equivalence:
                    items.append('{}={}'.format(printer(dim), printer(task_dim)))
        if op.share_map:
            items.append('share_map={}'.format(op.share_map))
        if len(items) > 0:
            return Text('# ') + doc_join(items, ', ')
        else:
            return Doc()

    param_docs = []
    for x in graph.inputs:
        name = printer.namer(x)
        param_docs.append(Text(name) + ': ' + get_tensor_sig(x))

    # head
    head_doc = 'Graph(' + doc_join(param_docs, ', ') + ')'

    for graph_input in graph.inputs:
        for dim in graph_input.shape:
            if isinstance(dim, SymbolVar):
                size_var_equivalence[dim] = dim

    # body
    body_doc = Doc()
    const_doc = Doc()
    for op in graph.nodes:
        # const inputs
        for x in op.inputs:
            if x not in printer.namer.obj_name:
                assert x.storage is not None
                const_doc += NewLine() + printer.namer.get_name(x, hint='c') + ' = Constant(' + get_tensor_sig(x) + ')'
        outputs = op.outputs
        if len(outputs) > 1:
            raise NotImplementedError()
        output: Tensor = outputs[0]
        line_doc = Doc()
        line_doc += printer.namer(output) + ': ' + get_tensor_sig(output) + ' = '
        items = []
        for x in op.inputs:
            items.append(printer.namer(x))
        for name, value in op.attrs.items():
            items.append(name + '=' + get_attr_repr(value))
        for op_out, task_out in zip(op.outputs, op.task.outputs):
            for a, b in zip(op_out.shape, task_out.shape):
                if isinstance(b, SymbolVar):
                    size_var_equivalence[a] = size_var_equivalence[b] if b in size_var_equivalence else b
        line_doc += op.name + '(' + doc_join(items, ', ') + ')  ' + get_comment(op)
        body_doc += NewLine() + line_doc
        if hidet.option.get_option('debug_show_verbose_flow_graph'):
            body_doc += (NewLine() + printer(op.task)).indent()

    # return statement
    body_doc += NewLine() + Text('return ') + doc_join([printer.namer(x) for x in graph.outputs], ', ')

    graph_doc = head_doc + '{' + const_doc.indent() + body_doc.indent() + NewLine() + '}'
    return str(graph_doc)
