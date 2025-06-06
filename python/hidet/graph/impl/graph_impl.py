from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict
import hashlib
import hidet.option
from hidet.ir.expr import Expr
from hidet.graph.tensor import Tensor
from hidet.graph.operator import Operator, SymbolVar
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
    # use dict for ordered set behaviour
    # ordering needed for deterministic node ordering
    all_nodes: Dict[Operator, bool] = {}

    def find_all_nodes(u: Operator):
        all_nodes[u] = True
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
    return share_map


def _get_tensor_sig(printer, size_var_equivalence, x: Tensor) -> Doc:
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


def _get_attr_repr(printer, value: Union[float, int, bool, str, list, tuple, FlowGraph]) -> Doc:
    if isinstance(value, (float, int, bool)):
        return Text(str(value))
    elif isinstance(value, str):
        return Text('"{}"'.format(value))
    elif isinstance(value, list):
        return '[' + doc_join([_get_attr_repr(printer, v) for v in value], ', ') + ']'
    elif isinstance(value, tuple):
        return '(' + doc_join([_get_attr_repr(printer, v) for v in value], ', ') + ')'
    elif isinstance(value, FlowGraph):
        return Text('FlowGraph({})'.format(', '.join(u.name for u in value.nodes)))
    elif isinstance(value, Expr):
        return printer(value)
    else:
        return Text(str(value))


def _get_comment(p, size_var_equivalence, op: Operator) -> Doc:
    items = []
    for out, task_output in zip(op.outputs, op.task.outputs):
        for dim, task_dim in zip(out.shape, task_output.shape):
            if isinstance(dim, SymbolVar) and task_dim not in size_var_equivalence:
                items.append('{}={}'.format(p(dim), p(task_dim)))
    if op.share_map:
        items.append('share_map={}'.format(op.share_map))
    if len(items) > 0:
        return Text('# ') + doc_join(items, ', ')
    else:
        return Doc()


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

    printer = IRPrinter()
    size_var_equivalence: Dict[SymbolVar, SymbolVar] = {}

    param_docs = []
    for x in graph.inputs:
        name = printer.namer(x)
        param_docs.append(Text(name) + ': ' + _get_tensor_sig(printer, size_var_equivalence, x))

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
                const_doc += (
                    NewLine()
                    + printer.namer.get_name(x, hint='c')
                    + ' = Constant('
                    + _get_tensor_sig(printer, size_var_equivalence, x)
                    + ')'
                    + ' # hash: {}'.format(hashlib.sha256(x.cpu().numpy().data.tobytes()).hexdigest()[:32])
                )
        outputs = op.outputs
        line_doc = Doc()
        for idx, output in enumerate(outputs):
            line_doc += printer.namer(output) + ': ' + _get_tensor_sig(printer, size_var_equivalence, output)
            if idx < len(outputs) - 1:
                line_doc += ', '
        line_doc += ' = '
        items = []
        for x in op.inputs:
            items.append(printer.namer(x))
        for name, value in op.attrs.items():
            items.append(name + '=' + _get_attr_repr(printer, value))
        for op_out, task_out in zip(op.outputs, op.task.outputs):
            for a, b in zip(op_out.shape, task_out.shape):
                if isinstance(b, SymbolVar):
                    size_var_equivalence[a] = size_var_equivalence[b] if b in size_var_equivalence else b
        line_doc += op.name + '(' + doc_join(items, ', ') + ')  ' + _get_comment(printer, size_var_equivalence, op)
        body_doc += NewLine() + line_doc
        if hidet.option.get_option('debug_show_verbose_flow_graph'):
            body_doc += (NewLine() + printer(op.task)).indent()

    # return statement
    body_doc += NewLine() + Text('return ') + doc_join([printer.namer(x) for x in graph.outputs], ', ')

    graph_doc = head_doc + '{' + const_doc.indent() + body_doc.indent() + NewLine() + '}'
    return str(graph_doc)


def draw_graph(graph: FlowGraph, filename: str, show_tensors: bool = False):
    """
    Draw the flow graph and save it to a DOT file.

    To visualize the .dot file, you can use
    https://dreampuf.github.io/GraphvizOnline/

    Parameters
    ----------
    graph: FlowGraph
        The flow graph to be visualized.
    filename: str
        The filename to save the graph visualization. If it ends with .dot, it will be used as is.
        Otherwise, .dot extension will be added.
    show_tensors: bool
        Whether to show tensor nodes in the graph.
    """
    import networkx as nx
    import os
    from hidet.ir.tools.printer import IRPrinter

    # Create directed graph and node mapping dictionary
    G = nx.DiGraph()
    printer = IRPrinter()
    tensor_to_id = {}

    # Add input nodes
    for i, tensor in enumerate(graph.inputs):
        tensor_id = f"input_{i}"
        tensor_to_id[tensor] = tensor_id
        G.add_node(tensor_id, label=f"Input {i}\n{tensor.shape}", node_type="input")

    # Add operator nodes and edges
    for i, op in enumerate(graph.nodes):
        op_id = f"op_{i}"

        # Create label with output shape
        output_shapes = [str(out.shape) for out in op.outputs]
        shape_str = output_shapes[0] if len(output_shapes) == 1 else ", ".join(output_shapes)
        G.add_node(op_id, label=f"{op.name}\n{shape_str}", node_type="operator")

        # Connect inputs
        for inp in op.inputs:
            if inp in tensor_to_id:
                src_id = tensor_to_id[inp]
                if src_id.startswith("input_"):
                    G.add_edge(src_id, op_id)
                elif show_tensors or src_id.startswith("op_"):
                    G.add_edge(src_id, op_id)

        # Add tensor nodes if needed
        for j, out in enumerate(op.outputs):
            tensor_id = f"tensor_{i}_{j}"
            tensor_to_id[out] = op_id  # Default: map to producing operator

            if show_tensors:
                tensor_label = f"{printer.namer(out)}\n{out.shape}"
                G.add_node(tensor_id, label=tensor_label, node_type="tensor")
                G.add_edge(op_id, tensor_id)
                tensor_to_id[out] = tensor_id  # Update to tensor node

    # Add output nodes
    for i, tensor in enumerate(graph.outputs):
        output_id = f"output_{i}"
        G.add_node(output_id, label=f"Output {i}", node_type="output")

        if tensor in tensor_to_id:
            src_id = tensor_to_id[tensor]
            if src_id.startswith("op_") or show_tensors:
                G.add_edge(src_id, output_id)

    # Set DOT filename and create directory if needed
    dot_filename = filename if filename.lower().endswith('.dot') else f"{os.path.splitext(filename)[0]}.dot"
    os.makedirs(os.path.dirname(os.path.abspath(dot_filename)), exist_ok=True)

    # Set node styles for DOT output
    node_styles = {
        'input': {'shape': 'ellipse', 'fillcolor': 'skyblue'},
        'operator': {'shape': 'box', 'fillcolor': 'lightgreen'},
        'tensor': {'shape': 'ellipse', 'fillcolor': 'lightyellow'},
        'output': {'shape': 'ellipse', 'fillcolor': 'salmon'},
        'matmul': {'shape': 'box', 'fillcolor': 'gold'},  # Special style for Matmul operators
    }

    # Apply styles to nodes
    for n, data in G.nodes(data=True):
        if 'node_type' in data:
            # Use special style for Matmul operators
            if data['node_type'] == 'operator' and 'label' in data and 'Matmul' in data['label']:
                style = node_styles['matmul']
            else:
                style = node_styles[data['node_type']]
            G.nodes[n].update({**style, 'style': 'filled'})

    # Save DOT file
    nx.nx_agraph.write_dot(G, dot_filename)

    # Add link to online renderer at the beginning of the .dot file
    with open(dot_filename, 'r') as f:
        dot_content = f.read()
    with open(dot_filename, 'w') as f:
        f.write("# You can use https://dreampuf.github.io/GraphvizOnline/ to visualize the graph\n\n" + dot_content)
