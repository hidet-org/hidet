from __future__ import annotations
from typing import List, Union, Dict, Set, Optional, Tuple
import os
import pickle
import warnings
from collections import defaultdict

import hidet.graph.operator
from hidet.graph.tensor import Tensor, empty_like
from hidet.graph.operator import Operator
from hidet.utils import tracer
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.utils.namer import Namer


class FlowGraph:
    """The computation graph representation.
    """
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
            return Text(x.dtype) + '[' + doc_join([str(v) for v in x.shape], ', ') + ']'

        def get_attr_repr(value: Union[float, int, bool, str, list, tuple]) -> Doc:
            from hidet.ir.expr import Constant
            if isinstance(value, (float, int, bool)):
                return Text(str(value))
            elif isinstance(value, str):
                return Text('"{}"'.format(value))
            elif isinstance(value, list):
                return '[' + doc_join([get_attr_repr(v) for v in value], ', ') + ']'
            elif isinstance(value, tuple):
                return '(' + doc_join([get_attr_repr(v) for v in value], ', ') + ')'
            # elif isinstance(value, Constant):
            #     return get_attr_repr(value.value)
            else:
                raise ValueError(value)

        param_docs = []
        for x in self.inputs:
            name = namer(x)
            param_docs.append(Text(name) + ': ' + get_tensor_sig(x))

        # head
        head_doc = 'Graph(' + doc_join(param_docs, ', ') + ')'

        # body
        body_doc = Doc()
        for op in self.nodes:
            # const inputs
            for x in op.inputs:
                if x not in namer.obj_name:
                    assert x.storage is not None
                    body_doc += NewLine() + namer.get_name(x, hint='c') + ' = ' + 'Constant(' + get_tensor_sig(x) + ')'
            outputs = op.outputs
            if len(outputs) > 1:
                raise NotImplementedError()
            output: Tensor = outputs[0]
            line_doc = Doc()
            line_doc += namer(output) + ' = '
            line_doc += op.name + ('*' if len(op.task.prologues) + len(op.task.epilogues) > 0 else '') + '('
            line_doc += doc_join([namer(x) for x in op.inputs], sep=', ')
            if op.attrs:
                line_doc += ', ' + doc_join([Text(name) + '=' + get_attr_repr(value) for name, value in op.attrs.items()], ', ')
            line_doc += ')'
            line_doc += '  # ' + get_tensor_sig(output)
            body_doc += NewLine() + line_doc

        # return statement
        body_doc += NewLine() + Text('return ') + doc_join([namer(x) for x in self.outputs], ', ')

        graph_doc = head_doc + '{' + body_doc.indent() + NewLine() + '}'
        return str(graph_doc)

    def build(self):
        tasks = []
        tunable_tasks = []
        task_keys = set()
        space_level = hidet.get_space_level()
        profile_config = hidet.get_profile_config()
        for node in self.nodes:
            if node.task_func is None:
                # if space_level == 0 or 'implement_cuda' not in node.task.__class__.__dict__:
                task_key = hash(str(node.task))
                if task_key in task_keys:
                    continue
                task_keys.add(task_key)
                if node.task.fast_implement(space_level):
                    tasks.append(node.task)
                else:
                    tunable_tasks.append(node.task)
        hidet.driver.build_batch_task(tasks, space_level, warmup=profile_config.warmup, number=profile_config.number, repeat=profile_config.repeat, parallel=True)
        # hidet.driver.build_batch_task(tasks, space_level, warmup=profile_config.warmup, number=profile_config.number, repeat=profile_config.repeat, parallel=False)
        hidet.driver.build_batch_task(tunable_tasks, space_level, warmup=profile_config.warmup, number=profile_config.number, repeat=profile_config.repeat, parallel=False)

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
        outputs = self.dummy_outputs()
        self.pure_forward(list(inputs), outputs)
        return outputs[0] if len(outputs) == 1 else outputs

    def pure_forward(self, inputs: List[Tensor], outputs: List[Tensor]):
        """Run the computation graph and store results to given tensors.

        Parameters
        ----------
        inputs: List[Tensor]
            The input tensors.

        outputs: List[Tensor]
            The output tensors to store the output results to.
        """
        for idx, tensor in enumerate(inputs):
            if tensor.storage is None:
                raise ValueError('Expect non-symbolic input tensors, got symbolic input {} ({}).'.format(idx, tensor.signature()))
        for idx, tensor in enumerate(outputs):
            if tensor.storage is None:
                raise ValueError('Expect non-symbolic output tensors, got symbolic output {} ({}).'.format(idx, tensor.signature()))
        if any(v is None for v in [self.inputs, self.nodes, self.usage_count]):
            self.update_nodes()
        self.build()

        usage_count = self.usage_count.copy()
        tensor_map: Dict[Tensor, Tensor] = {}
        for st, at in zip(self.inputs, inputs):
            tensor_map[st] = at
        for st, at in zip(self.outputs, outputs):
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

            # prepare node outputs
            node_outputs = node.dummy_outputs()
            for i, symbolic_output in enumerate(node.outputs):
                if symbolic_output in tensor_map:   # the output is a graph output
                    node_outputs[i] = tensor_map[symbolic_output]
                else:
                    tensor_map[symbolic_output] = node_outputs[i]

            # run node
            node.pure_run(node_inputs, node_outputs)

    def dummy_inputs(self) -> List[Tensor]:
        inputs = []
        for symbolic_input in self.inputs:
            if symbolic_input.dtype in ['float32', 'float16', 'bfloat16']:
                inputs.append(empty_like(symbolic_input))
            else:
                raise ValueError('Can not generate dummy input for tensor {}'.format(symbolic_input.signature()))
        return inputs

    def dummy_outputs(self) -> List[Tensor]:
        return [empty_like(tensor) if tensor.storage is None else tensor for tensor in self.outputs]

    def save(self, fname: str):
        # before save, clear the packed func cache because ctypes object can not be pickled
        for node in self.nodes:
            node.task_func = None
        self.usage_count, self.nodes = None, None

        dirname = os.path.dirname(fname)
        os.makedirs(dirname, exist_ok=True)
        # save to a temporary file first, in case pickle fails.
        with open(fname + '.temp', 'wb') as f:
            pickle.dump(self, f)
        os.rename(fname + '.temp', fname)

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

    def cuda_graph(self):
        """Create a CudaGraph from FlowGraph.

        Returns
        -------
        ret: hidet.runtime.CudaGraph
            The created cuda graph.
        """
        from hidet.runtime.cuda_graph import create_cuda_graph
        return create_cuda_graph(self)

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
