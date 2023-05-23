# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Union

import hidet.option
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.graph.ir.flow_graph import FlowGraph, Operator, Tensor


def flow_graph_as_text(graph: FlowGraph) -> str:
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

        return Text(x.dtype.name) + '[' + doc_join(shape_items, ', ') + ']'

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
