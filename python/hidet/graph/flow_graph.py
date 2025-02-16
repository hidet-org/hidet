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

import hidet.graph.operator
import hidet.cuda
from hidet.cuda.graph import CudaGraphCreationError
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

    def debug(
        self,
        output_dir: Union[str, bytes, os.PathLike] = './outs/debug',
        *,
        print_summary: bool = False,
        dump_outputs: bool = False,
        dump_op: bool = False,
    ):
        """
        Dump parts of the computation graph for debugging. By default, outputs a summary
        of the operators and a Netron graph for viewing.

        Parameters
        ----------
        output_dir: str, bytes, or PathLike
            The directory to store the outputs (default: ./outs/debug)
        print_summary: bool
            Whether to print the summary to stdout (default: false)
        dump_outputs: bool
            Whether to dump outputs of operators (default: false)
        dump_op: bool
            Whether to dump the operator definition (default: false)
        """
        from .graph_utils.instruments import GraphForwardDebugInstrument

        self.instruments.append(
            GraphForwardDebugInstrument(
                output_dir=output_dir, print_summary=print_summary, dump_outputs=dump_outputs, dump_op=dump_op
            )
        )

    def benchmark(self, output_dir='./outs/benchmark', print_summary: bool = False, warmup=3, number=10, repeat=3):
        from .graph_utils.instruments import GraphForwardBenchmarkInstrument

        self.instruments.append(GraphForwardBenchmarkInstrument(output_dir, print_summary, warmup, number, repeat))


def forward_context() -> GraphForwardContext:
    return GraphForwardContext()


class FlowGraph:
    """
    The computation graph representation.

    Attributes
    ----------
    outputs: List[Tensor]
        The output tensors of the computation graph.

    inputs: Optional[List[Tensor]]
        The input tensors of the computation graph.

    """

    def __init__(self, outputs: Sequence[Tensor], inputs: Optional[Sequence[Tensor]]):
        self.outputs: List[Tensor] = list(outputs)
        self.inputs: Optional[List[Tensor]] = list(inputs) if inputs else None
        self._share_map: Optional[Dict[int, int]] = None
        self._nodes: Optional[List[Operator]] = None
        self._usage_count: Optional[Dict[Tensor, int]] = None

        if self.inputs is None:
            # analyze the graph to get the inputs, when the inputs are not given, there should be only one input
            # when there are multiple inputs, it is mandatory to specify the "inputs" argument explicitly to avoid
            # ambiguity in the order of inputs
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
        from .impl.graph_impl import graph_as_text

        return graph_as_text(self)

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

    @property
    def share_map(self) -> Dict[int, int]:
        """
        If an output tensor of the graph shares the memory with an input tensor of the graph, we should record the
        information in this attribute. For example, `share_map = {0: 0, 1: 2}` means that the output tensor 0 shares
        the memory with input tensor 0, and output tensor 1 shares the memory with input tensor 2 of the graph.
        The output tensor does not allow sharing memory with intermediate tensors in the graph.
        """
        if self._share_map is None:
            from hidet.graph.impl.graph_impl import graph_analyze_share_map

            self._share_map = graph_analyze_share_map(self)
        return self._share_map.copy()

    def invalid_cache(self):
        self._nodes = None
        self._usage_count = None

    def _build_nodes(self):
        tasks: List[Tuple[Task, str]] = []
        task_keys = set()
        for node in self.nodes:
            if node._compiled_task is None:
                task_key = hash(str(node.task))
                if task_key in task_keys:
                    continue
                task_keys.add(task_key)
                tasks.append((node.task, node.build_target))

        hidet.drivers.build_task_batch(tasks)

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
            node_outputs = node.compiled_task.run_async(node_inputs)
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

    def update_nodes(self, weight_tensors=None):
        from hidet.graph.impl.graph_impl import graph_analyze

        inputs = self.inputs if self.inputs else []
        weight_tensors = weight_tensors if weight_tensors else []
        free_vars, self._nodes, self._usage_count = graph_analyze(self.outputs, stop_tensors=inputs + weight_tensors)

        if self.inputs:
            non_bound_free_vars: Set[Tensor] = set(free_vars)
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
        self, warmup=25, repeat=100, dummy_inputs: Optional[Sequence[Tensor]] = None
    ) -> Union[float, List[float]]:
        """Measure the latency of the flow graph.

        Parameters
        ----------
        warmup: int
            The number of warmup runs.

        repeat: int
            The number of times to repeat the measurement.

        dummy_inputs: Optional[Sequence[Tensor]]
            The dummy inputs to run the flow graph. If not given, automatic generated dummy inputs would be used.

        Returns
        -------
        ret: Union[float, List[float]]
            The measured latency in milliseconds.
        """
        from hidet.utils.benchmark import do_bench

        if dummy_inputs is None:
            dummy_inputs = self.dummy_inputs()

        # return the median
        return do_bench(lambda: self.forward(dummy_inputs), warmup=warmup, rep=repeat)[1]

    def vcuda_(self) -> None:
        """
        casts the flow graph object to vcuda device in place
        """
        from hidet.runtime.device import instantiate_device, Device

        for x in self.inputs:
            if not x.device.is_cuda():
                raise ValueError("Inputs must be on cuda device")
            x.vcuda_()

        for node in self.nodes:
            if 'device' in node.attrs:
                dev = instantiate_device(node.attrs['device'])
                if dev.is_cuda():
                    dev = Device('vcuda', dev.id)
                node.attrs['device'] = dev
            for inp in node.inputs:
                if inp.device.is_cuda():
                    inp.vcuda_()
            for outp in node.outputs:
                if outp.device.is_cuda():
                    outp.vcuda_()

    def cuda_(self) -> None:
        """
        casts the flow graph object from vcuda device in place
        """
        from hidet.runtime.device import instantiate_device, Device

        for x in self.inputs:
            if not x.device.is_vcuda():
                raise ValueError("Inputs must be on vcuda device")
            x.cuda_()

        for node in self.nodes:
            if 'device' in node.attrs:
                dev = instantiate_device(node.attrs['device'])
                if dev.is_vcuda():
                    dev = Device('cuda', dev.id)
                node.attrs['device'] = dev
            for inp in node.inputs:
                if inp.device.is_vcuda():
                    inp.cuda_()
            for outp in node.outputs:
                if outp.device.is_vcuda():
                    outp.cuda_()

    def vhip_(self) -> None:
        """
        casts the flow graph object to vhip device in place
        """
        from hidet.runtime.device import instantiate_device, Device

        for x in self.inputs:
            if not x.device.is_hip():
                raise ValueError("Inputs must be on hip device")
            x.vhip_()

        for node in self.nodes:
            if 'device' in node.attrs:
                dev = instantiate_device(node.attrs['device'])
                if dev.is_hip():
                    dev = Device('vhip', dev.id)
                node.attrs['device'] = dev
            for inp in node.inputs:
                if inp.device.is_hip():
                    inp.vhip_()
            for outp in node.outputs:
                if outp.device.is_hip():
                    outp.vhip_()

    def hip_(self) -> None:
        """
        casts the flow graph object from vhip device in place
        """
        from hidet.runtime.device import instantiate_device, Device

        for x in self.inputs:
            if not x.device.is_vhip():
                raise ValueError("Inputs must be on vhip device")
            x.hip_()

        for node in self.nodes:
            if 'device' in node.attrs:
                dev = instantiate_device(node.attrs['device'])
                if dev.is_vhip():
                    dev = Device('hip', dev.id)
                node.attrs['device'] = dev
            for inp in node.inputs:
                if inp.device.is_vhip():
                    inp.hip_()
            for outp in node.outputs:
                if outp.device.is_vhip():
                    outp.hip_()


def trace_from(
    tensor: Union[Tensor, List[Tensor]], inputs: Optional[Union[Tensor, List[Tensor]]] = None, weight_tensors=None
) -> FlowGraph:
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
    weight_tensors: Optional, List[Tensor]
        Torch (from version 2.5.1) compile treats weight tensors as inputs.
        To treat weights as constant tensors in FlowGraph weights should be marked as stop tensors.

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
    return FlowGraph(outputs, inputs).update_nodes(weight_tensors)


def save_graph(graph: FlowGraph, fname: str):
    graph.save(fname)


def load_graph(fname: str) -> FlowGraph:
    return FlowGraph.load(fname)
