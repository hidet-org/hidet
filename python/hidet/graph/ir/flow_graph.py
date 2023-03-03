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
# pylint: disable=protected-access
from __future__ import annotations
from typing import List, Union, Dict, Set, Optional, Tuple, Sequence
import logging
import os
import pickle
from collections import defaultdict

import hidet.graph.operator
import hidet.cuda
from hidet import option
from hidet.graph.tensor import Tensor, zeros_like, randn_like
from hidet.graph.operator import Operator
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.utils.namer import Namer

logger = logging.getLogger(__name__)


class GraphForwardInstrument:
    def before_graph(self, graph: FlowGraph, inputs: List[Tensor]) -> None:
        pass

    def after_graph(self, graph: FlowGraph, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        pass

    def before_operator(self, op: Operator, inputs: List[Tensor]) -> None:
        pass

    def after_operator(self, op: Operator, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        pass


class GraphForwardContext:
    _stack: List[GraphForwardContext] = []

    def __init__(self):
        self.instruments: List[GraphForwardInstrument] = []

    def __enter__(self):
        GraphForwardContext._stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        GraphForwardContext._stack.pop()

    @staticmethod
    def current() -> GraphForwardContext:
        if len(GraphForwardContext._stack) == 0:
            GraphForwardContext._stack.append(GraphForwardContext())
        return GraphForwardContext._stack[-1]

    def _trigger_before_graph(self, graph: FlowGraph, inputs: List[Tensor]) -> None:
        for instrument in self.instruments:
            instrument.before_graph(graph, inputs)

    def _trigger_after_graph(self, graph: FlowGraph, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        for instrument in self.instruments:
            instrument.after_graph(graph, inputs, outputs)

    def _trigger_before_operator(self, op: Operator, inputs: List[Tensor]) -> None:
        for instrument in self.instruments:
            instrument.before_operator(op, inputs)

    def _trigger_after_operator(self, op: Operator, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        for instrument in self.instruments:
            instrument.after_operator(op, inputs, outputs)

    def append_instrument(self, instrument: GraphForwardInstrument):
        self.instruments.append(instrument)

    def debug(self, output_dir='./outs/debug', print_summary: bool = False):
        from .flow_graph_impl import GraphForwardDebugInstrument

        self.instruments.append(GraphForwardDebugInstrument(output_dir, print_summary))

    def benchmark(self, output_dir='./outs/benchmark', print_summary: bool = False, warmup=3, number=10, repeat=3):
        from .flow_graph_impl import GraphForwardBenchmarkInstrument

        self.instruments.append(GraphForwardBenchmarkInstrument(output_dir, print_summary, warmup, number, repeat))


def forward_context() -> GraphForwardContext:
    return GraphForwardContext()


class FlowGraph:
    """The computation graph representation."""

    def __init__(self, outputs: List[Tensor], inputs=None, nodes=None):
        self.outputs: List[Tensor] = outputs
        self.inputs: Optional[List[Tensor]] = inputs
        self.nodes: Optional[List[Operator]] = nodes
        self.usage_count: Optional[Dict[Tensor, int]] = None

    def __call__(self, *inputs: Tensor) -> Union[List[Tensor], Tensor]:
        """Run the computation graph.
        See Also :func:`FlowGraph.forward`.
        """
        return self.forward(*inputs)

    def __str__(self):
        if any(v is None for v in [self.inputs, self.nodes, self.usage_count]):
            self.update_nodes()
        namer = Namer()

        def get_tensor_sig(x: Tensor) -> Doc:
            return Text(x.dtype.name) + '[' + doc_join([str(v) for v in x.shape], ', ') + ']'

        def get_attr_repr(value: Union[float, int, bool, str, list, tuple]) -> Doc:
            if isinstance(value, (float, int, bool)):
                return Text(str(value))
            elif isinstance(value, str):
                return Text('"{}"'.format(value))
            elif isinstance(value, list):
                return '[' + doc_join([get_attr_repr(v) for v in value], ', ') + ']'
            elif isinstance(value, tuple):
                return '(' + doc_join([get_attr_repr(v) for v in value], ', ') + ')'
            else:
                return Text(str(value))

        param_docs = []
        for x in self.inputs:
            name = namer(x)
            param_docs.append(Text(name) + ': ' + get_tensor_sig(x))

        # head
        head_doc = 'Graph(' + doc_join(param_docs, ', ') + ')'

        # body
        body_doc = Doc()
        const_doc = Doc()
        for op in self.nodes:
            # const inputs
            for x in op.inputs:
                if x not in namer.obj_name:
                    assert x.storage is not None
                    const_doc += NewLine() + namer.get_name(x, hint='c') + ' = Constant(' + get_tensor_sig(x) + ')'
            outputs = op.outputs
            if len(outputs) > 1:
                raise NotImplementedError()
            output: Tensor = outputs[0]
            line_doc = Doc()
            line_doc += namer(output) + ': ' + get_tensor_sig(output) + ' = '
            line_doc += op.name + ('*' if len(op.task.task_graph.nodes) > 1 else '') + '('
            line_doc += doc_join([namer(x) for x in op.inputs], sep=', ')
            if op.attrs:
                line_doc += ', ' + doc_join(
                    [Text(name) + '=' + get_attr_repr(value) for name, value in op.attrs.items()], ', '
                )
            line_doc += ')  '
            body_doc += NewLine() + line_doc

        # return statement
        body_doc += NewLine() + Text('return ') + doc_join([namer(x) for x in self.outputs], ', ')

        graph_doc = head_doc + '{' + const_doc.indent() + body_doc.indent() + NewLine() + '}'
        return str(graph_doc)

    def build(self):
        tasks = []
        tunable_tasks = []
        task_keys = set()
        search_space = hidet.option.get_option('search_space')
        for node in self.nodes:
            if node.task_func is None:
                task_key = hash(str(node.task))
                if task_key in task_keys:
                    continue
                task_keys.add(task_key)
                if search_space == 0 or 'implement_cuda' not in node.task.__class__.__dict__:
                    tasks.append(node.task)
                else:
                    tunable_tasks.append(node.task)

        hidet.driver.build_task_batch(tasks)

        with option.context():
            hidet.option.parallel_build(False)
            hidet.driver.build_task_batch(tunable_tasks)  # build tunable tasks one by one

    def forward(self, *inputs: Tensor) -> Union[List[Tensor], Tensor]:
        """Run the computation graph.

        Parameters
        ----------
        *inputs: Tensor
            The input tensors. They should be consistent with the symbolic inputs
            of the computation graph.

        Returns
        -------
        output: Union[List[Tensor], Tensor]
            If there is only one output, it is returned directly. Otherwise, a list
            of output tensors are returned.
        """
        inputs: List[Tensor] = list(inputs)

        for idx, tensor in enumerate(inputs):
            if tensor.storage is None:
                msg = 'Expect non-symbolic input tensors, got symbolic input {} ({}).'.format(idx, tensor.signature())
                raise ValueError(msg)
        if any(v is None for v in [self.inputs, self.nodes, self.usage_count]):
            self.update_nodes()
        self.build()

        GraphForwardContext.current()._trigger_before_graph(self, inputs)

        usage_count = self.usage_count.copy()
        tensor_map: Dict[Tensor, Tensor] = {}
        for st, at in zip(self.inputs, inputs):
            tensor_map[st] = at

        num_operators = len(self.nodes)
        for idx, node in enumerate(self.nodes):
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
            GraphForwardContext.current()._trigger_before_operator(node, node_inputs)
            logger.debug('[%4d/%d] run operator %s', idx, num_operators, node.name)
            logger.debug('   inputs: %s', [x.signature() for x in node_inputs])
            node_outputs = node.imperative_run(node_inputs)
            logger.debug('  outputs: %s', [x.signature() for x in node_outputs])
            GraphForwardContext.current()._trigger_after_operator(node, node_inputs, node_outputs)

            # update map
            for node_output, symbolic_output in zip(node_outputs, node.outputs):
                tensor_map[symbolic_output] = node_output

        outputs = []
        for graph_output in self.outputs:
            if graph_output in tensor_map:
                outputs.append(tensor_map[graph_output])
            elif graph_output.storage is not None:
                outputs.append(graph_output)  # constant output, not the graph input or produced by any operator
            else:
                raise RuntimeError('Graph output {} is not produced by any operator.'.format(graph_output.signature()))

        GraphForwardContext.current()._trigger_after_graph(self, inputs, outputs)
        return outputs[0] if len(outputs) == 1 else outputs

    def dummy_inputs(self) -> List[Tensor]:
        inputs = []
        for symbolic_input in self.inputs:
            if symbolic_input.dtype.is_integer():
                inputs.append(zeros_like(symbolic_input))
            elif symbolic_input.dtype.is_float():
                inputs.append(randn_like(symbolic_input))
            else:
                assert False
        return inputs

    def save(self, model_file: str):
        """Save the flow graph to a file.

        Parameters
        ----------
        model_file: str
            The model file to store the flow graph.
        """
        # before save, clear the packed func cache because ctypes object can not be pickled
        for node in self.nodes:
            node.task_func = None
        self.usage_count, self.nodes = None, None

        dirname = os.path.dirname(model_file)
        os.makedirs(dirname, exist_ok=True)
        # save to a temporary file first, in case pickle fails.
        with open(model_file + '.temp', 'wb') as f:
            pickle.dump(self, f)
        os.rename(model_file + '.temp', model_file)

    @staticmethod
    def load(model_file: str) -> FlowGraph:
        """Load a flow graph from a file.

        Parameters
        ----------
        model_file: str
            The path to the flow graph.

        Returns
        -------
        ret: FlowGraph
            The loaded flow graph.
        """
        with open(model_file, 'rb') as f:
            ret = pickle.load(f)
        if not isinstance(ret, FlowGraph):
            raise TypeError('Expect to load FlowGraph, got {}'.format(type(ret)))
        ret.update_nodes()
        return ret

    def update_nodes(self):
        free_vars, self.nodes, self.usage_count = self._analyze(self.outputs)
        if self.inputs:
            non_bound_free_vars: Set[Tensor] = set(free_vars) - set(self.inputs)
            if len(non_bound_free_vars) > 0:
                msg = ['There is free variable(s) not given in inputs:']
                for v in non_bound_free_vars:
                    msg.append('  {}'.format(v.signature()))
                raise ValueError('\n'.join(msg))
        else:
            if len(free_vars) > 1:
                raise ValueError(
                    f'The traced graph has found {len(free_vars)} free varaibles. '
                    f'When there are multiple free '
                    f'variables, it is mandatory to specify the "inputs" argument explicitly when calling '
                    f'hidet.trace_from(...):\n'
                    '    hidet.trace_from(..., inputs=[tensor1, tensor2, ...])\n'
                )
            self.inputs = free_vars
        return self

    def cuda_graph(self):
        """Create a CudaGraph from FlowGraph.

        Returns
        -------
        ret: hidet.cuda.graph.CudaGraph
            The created cuda graph.
        """
        from hidet.cuda.graph import CudaGraph

        return CudaGraph(self)

    def latency(
        self, warmup=1, number=3, repeat=3, median=True, dummy_inputs: Optional[Sequence[Tensor]] = None
    ) -> Union[float, List[float]]:
        """Measure the latency of the flow graph.

        Parameters
        ----------
        warmup: int
            The number of warmup runs.

        number: int
            The number of runs to measure the latency.

        repeat: int
            The number of times to repeat the measurement.

        median: bool
            Whether to return the median latency.

        dummy_inputs: Optional[Sequence[Tensor]]
            The dummy inputs to run the flow graph. If not given, automatic generated dummy inputs would be used.

        Returns
        -------
        ret: Union[float, List[float]]
            The measured latency in milliseconds.
        """
        import time
        import numpy as np

        if dummy_inputs is None:
            dummy_inputs = self.dummy_inputs()
        for _ in range(warmup):
            self.forward(*dummy_inputs)
        results = []
        for _ in range(repeat):
            hidet.cuda.synchronize()
            t1 = time.time()
            for _ in range(number):
                self.forward(*dummy_inputs)
            hidet.cuda.synchronize()
            t2 = time.time()
            results.append((t2 - t1) * 1000 / number)
        if median:
            return float(np.median(results))
        else:
            return results

    @staticmethod
    def _analyze(outputs: List[Tensor]) -> Tuple[List[Tensor], List[Operator], Dict[Tensor, int]]:
        """
        Analyze the implicit flow graph by backwards traversing the graph from given outputs.

        Parameters
        ----------
        outputs: List[Tensor]
            The outputs of the flow graph to traversing from.

        Returns
        -------
        free_vars, nodes, usage_count: Tuple[List[Tensor], List[Operator], Dict[Tensor, int]]
            The free variables, nodes and usage count of the flow graph.

            The free variables are the free symbolic tensors that are not produced by any operators and do not contain
            the non-None storage attribute.

            The nodes are the operators that are used to produce the outputs, in topological order.

            The usage count contains the number of times each tensor is used.
        """
        free_vars = []
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
                    if it.storage is None and all(it is not v for v in free_vars):
                        # input
                        free_vars.append(it)
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

        return free_vars, nodes, usage_count


def trace_from(tensor: Union[Tensor, List[Tensor]], inputs: Optional[Union[Tensor, List[Tensor]]] = None) -> FlowGraph:
    """
    Trace the flow graph given the output tensor(s).

    Each :class:`hidet.graph.Tensor` has an attribute :class:`hidet.graph.Tensor.trace` which indicates how the tensor
    is generated. If the tensor is generated by an operator with symbolic input(s), the tensor itself is also symbolic.
    And the tensor will have a reference to the operator that generates it. The reference is stored in this attribute.

    What this function does is to walk through the trace of the given tensor(s) and construct a flow graph.

    When there are multiple symbol inputs, it is mandatory to specify the "inputs" argument explicitly to avoid
    ambiguity.

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
        assert all(isinstance(v, Tensor) for v in outputs)
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
