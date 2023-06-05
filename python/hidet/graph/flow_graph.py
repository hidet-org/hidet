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
from hidet.cuda.graph import CudaGraphCreationError
from hidet import option
from hidet.ir.expr import is_constant
from hidet.ir.task import Task
from hidet.graph.tensor import Tensor, zeros_like, randn_like
from hidet.graph.operator import Operator, SymbolVar

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

    @staticmethod
    def _before_graph(graph: FlowGraph, inputs: List[Tensor]) -> None:
        ctx = GraphForwardContext.current()
        for instrument in ctx.instruments:
            instrument.before_graph(graph, inputs)

    @staticmethod
    def _after_graph(graph: FlowGraph, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        ctx = GraphForwardContext.current()
        for instrument in ctx.instruments:
            instrument.after_graph(graph, inputs, outputs)

    @staticmethod
    def _before_operator(op: Operator, inputs: List[Tensor]) -> None:
        ctx = GraphForwardContext.current()
        for instrument in ctx.instruments:
            instrument.before_operator(op, inputs)

    @staticmethod
    def _after_operator(op: Operator, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        ctx = GraphForwardContext.current()
        for instrument in ctx.instruments:
            instrument.after_operator(op, inputs, outputs)

    def append_instrument(self, instrument: GraphForwardInstrument):
        self.instruments.append(instrument)

    def debug(self, output_dir='./outs/debug', print_summary: bool = False, dump_outputs: bool = False):
        from .graph_utils.instruments import GraphForwardDebugInstrument

        self.instruments.append(GraphForwardDebugInstrument(output_dir, print_summary, dump_outputs))

    def benchmark(self, output_dir='./outs/benchmark', print_summary: bool = False, warmup=3, number=10, repeat=3):
        from .graph_utils.instruments import GraphForwardBenchmarkInstrument

        self.instruments.append(GraphForwardBenchmarkInstrument(output_dir, print_summary, warmup, number, repeat))


def forward_context() -> GraphForwardContext:
    return GraphForwardContext()


class FlowGraph:
    """The computation graph representation."""

    def __init__(self, outputs: Sequence[Tensor], inputs: Optional[Sequence[Tensor]] = None, nodes=None):
        self.outputs: List[Tensor] = list(outputs)
        self.inputs: Optional[List[Tensor]] = list(inputs) if inputs is not None else None
        self._nodes: Optional[List[Operator]] = nodes
        self._usage_count: Optional[Dict[Tensor, int]] = None
        self.update_nodes()

    def __call__(self, *inputs: Tensor) -> Union[List[Tensor], Tensor]:
        """
        Run the computation graph.

        Parameters
        ----------
        inputs : Sequence[Tensor]
            The input tensors.

        Returns
        -------
        ret: Union[List[Tensor], Tensor]
            The output tensors. If there is only one output, return it directly.
        """
        outputs = self.forward(list(inputs))
        return outputs[0] if len(outputs) == 1 else outputs

    def __str__(self):
        from .graph_utils import flow_graph_as_text

        return flow_graph_as_text(self)

    @property
    def nodes(self) -> List[Operator]:
        """The list of operators in the computation graph."""
        if self._nodes is None:
            self.update_nodes()
        return self._nodes

    @property
    def usage_count(self) -> Dict[Tensor, int]:
        """The usage count of each tensor in the computation graph."""
        if self._usage_count is None:
            self.update_nodes()
        return self._usage_count.copy()

    def invalid_cache(self):
        self._nodes = None
        self._usage_count = None

    def _build_nodes(self):
        tasks: List[Tuple[Task, str]] = []
        tunable_tasks: List[Tuple[Task, str]] = []
        task_keys = set()
        search_space = hidet.option.get_option('search_space')
        for node in self.nodes:
            if node._compiled_task is None:
                task_key = hash(str(node.task))
                if task_key in task_keys:
                    continue
                task_keys.add(task_key)
                if search_space == 0 or all(
                    method not in node.task.__class__.__dict__
                    for method in ['implement_cuda', 'implement_cpu', 'implement']
                ):
                    tasks.append((node.task, node.build_target))
                else:
                    tunable_tasks.append((node.task, node.build_target))

        hidet.drivers.build_task_batch(tasks)

        with option.context():
            hidet.option.parallel_build(False)
            hidet.drivers.build_task_batch(tunable_tasks)  # build tunable tasks one by one

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """Run the computation graph.

        Parameters
        ----------
        inputs: List[Tensor]
            The input tensors. They should be consistent with the symbolic inputs
            of the computation graph.

        Returns
        -------
        output: List[Tensor]
            The output tensors of the computation graph.
        """
        from hidet.ffi import runtime_api

        inputs: List[Tensor] = list(inputs)

        # the input tensors should be non-symbolic
        for idx, tensor in enumerate(inputs):
            if tensor.storage is None:
                msg = 'Expect non-symbolic input tensors, got symbolic input {} ({}).'.format(idx, tensor.signature())
                raise ValueError(msg)

        # build the kernel for each operator in the graph
        self._build_nodes()

        # set the symbol values
        for expect_input, actual_input in zip(self.inputs, inputs):
            if expect_input.device != actual_input.device:
                raise ValueError(
                    'Expect input {} to have device {}, got {}.'.format(
                        expect_input, expect_input.device, actual_input.device
                    )
                )
            for expect_dim, actual_dim in zip(expect_input.shape, actual_input.shape):
                if isinstance(expect_dim, SymbolVar):
                    runtime_api.set_symbol_value(expect_dim.name, int(actual_dim))
                else:
                    assert is_constant(actual_dim, expect_dim) and expect_dim == actual_dim

        GraphForwardContext._before_graph(self, inputs)

        # count the usage of each tensor. We use this count to determine whether
        # a tensor should be freed after running an operator.
        usage_count = self.usage_count.copy()
        tensor_map: Dict[Tensor, Tensor] = {}  # symbolic tensor -> actual tensor during the forward process
        for st, at in zip(self.inputs, inputs):
            tensor_map[st] = at

        # run each operator in the graph in a topological order
        for idx, node in enumerate(self.nodes):
            # prepare node inputs
            node_inputs = []
            for node_input in node.inputs:
                if node_input.storage is None:
                    # symbolic input
                    node_inputs.append(tensor_map[node_input])
                    usage_count[node_input] -= 1
                    if usage_count[node_input] == 0:  # this temporary tensor is no longer needed
                        # free the memory
                        del tensor_map[node_input]
                else:
                    # constant input
                    node_inputs.append(node_input)
            node_inputs = node_inputs[: len(node.inputs)]

            # run node
            GraphForwardContext._before_operator(node, node_inputs)
            logger.debug('[%4d/%d] run operator %s, %s', idx, len(self.nodes), node.name, node.task)
            logger.debug('   inputs: %s', [x.signature() for x in node_inputs])
            node_outputs = node.imperative_run(node_inputs)
            logger.debug('  outputs: %s', [x.signature() for x in node_outputs])
            GraphForwardContext._after_operator(node, node_inputs, node_outputs)

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

        GraphForwardContext._after_graph(self, inputs, outputs)
        return outputs

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
            node._compiled_task = None
        self._usage_count, self._nodes = None, None

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
        return ret

    def update_nodes(self):
        free_vars, self._nodes, self._usage_count = self._analyze(self.outputs)
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

    def build(self, *, space=0):
        """
        Build the flow graph to a compiled model (hidet.runtime.CompiledModel).

        Parameters
        ----------
        space: int
            The space to allocate for the compiled model. Candidates are 0, 1 and 2.
            Space 0 means each operator will be compiled with the default schedule. Space 1 means each operator will be
            compiled with a small set of schedules. Space 2 means each operator will be compiled with a large set of
            schedules. The larger the space, the more schedules will be tried, and the better the performance will be,
            with the cost of longer compilation and tuning time.

        Returns
        -------
        ret: hidet.runtime.CompiledGraph
            The compiled model.
        """
        from hidet.drivers.build_graph import build_flow_graph

        return build_flow_graph(self, space=space)

    def cuda_graph(self):
        """Create a CudaGraph from FlowGraph.

        Returns
        -------
        ret: hidet.cuda.graph.CudaGraph
            The created cuda graph.
        """
        from hidet.cuda.graph import CudaGraph

        for x in self.inputs:
            if not x.device.is_cuda():
                raise CudaGraphCreationError(
                    'FlowGraph.cuda_graph() only supports cuda inputs, got {}'.format(x.signature())
                )
            for d in x.shape:
                if not isinstance(d, int):
                    raise CudaGraphCreationError(
                        'FlowGraph.cuda_graph() only supports inputs with static shape, got {}'.format(x.signature())
                    )

        def f_create_inputs() -> List[Tensor]:
            return self.dummy_inputs()

        def f_run(inputs: List[Tensor]) -> List[Tensor]:
            return self.forward(inputs)

        return CudaGraph(f_create_inputs, f_run, ref_objs=[self])

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
                self.forward(dummy_inputs)
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
            for x in u.inputs:
                if x.op is None:
                    continue
                v: Operator = x.op
                if v not in all_nodes:
                    find_all_nodes(v)

        for ot in outputs:
            if ot.trace:
                find_all_nodes(ot.op)

        # topological sort
        out_degree: Dict[Operator, int] = {u: 0 for u in all_nodes}
        for u in all_nodes:
            for it in u.inputs:
                if it.op is None or it.op not in all_nodes:
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
