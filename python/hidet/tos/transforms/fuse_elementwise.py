from typing import Tuple, Dict, List, Optional, Union

import hidet.implement
from hidet.implement import Implementer
from hidet.ir.type import TypeNode
from hidet.ir.task import Task
from hidet.tos.ir.graph import FlowGraph, Operator, Tensor
from hidet.tos.transforms.base import GraphPass
from .common import analyze_usage, graph_collect
from hidet.ir.dialects.compute import TensorCompute, ReduceCompute, TensorInput
from hidet.ir import functors
from hidet.ir.functors import collect, rewrite
from hidet import tos


def trim_fuse_head(name: str):
    return name[5:] if name.startswith('Fused') else name


class FuseElementwisePass(GraphPass):
    max_num_fuses = 1000

    def __init__(self):
        super().__init__()
        self.task_resolve_cache: Dict[Task, Optional[Implementer]] = {}

    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        graph = tos.ir.functors.clone(graph)
        for t in range(self.max_num_fuses):
            updated, graph = self.try_fuse(graph)
            if not updated:
                return graph
        print('Exceeded maximum number of fuses {}, stop early.'.format(self.max_num_fuses))
        return graph

    def try_fuse(self, graph: FlowGraph) -> Tuple[bool, FlowGraph]:
        usage: Dict[Tensor, List[Tuple[Optional[Operator], int]]] = analyze_usage(graph)
        nodes = graph_collect(graph, Operator)
        for node in nodes:
            if self.is_elementwise(node):
                non_const_inputs = [tensor for tensor in node.inputs if tensor.trace is not None]
                if len(non_const_inputs) != 1:
                    continue
                x = non_const_inputs[0]
                if len(usage[x]) > 1:
                    continue
                prior_node = x.trace[0]
                op = self.fuse_nodes(nodes=[prior_node, node])
                if op is None:
                    continue
                for idx, out_tensor in enumerate(op.outputs):
                    out_tensor.trace = (op, idx)
                return True, graph
        return False, graph

    def is_elementwise(self, op: Operator) -> bool:
        task = op.task
        if not isinstance(task.compute, TensorCompute):
            return False
        value = task.compute.value
        if len(collect(value, (ReduceCompute, TensorCompute))) > 0:
            return False
        return True

    def fuse_nodes(self, nodes: List[Operator]) -> Optional[Operator]:
        if any(len(node.outputs) > 1 for node in nodes):
            return None
        # compute the inputs and outputs of fused operator
        all_inputs, all_outputs = [], []
        for node in nodes:
            all_inputs.extend(node.inputs)
            all_outputs.extend(node.outputs)
        input_tensors = [tensor for tensor in all_inputs if tensor not in all_outputs]
        output_tensors = [tensor for tensor in all_outputs if tensor not in all_inputs]
        if len(output_tensors) > 1:
            return None
        # construct fused task
        tensor_map: Dict[Tensor, Union[TensorInput, TensorCompute]] = {}
        tensor_type_map: Dict[Tensor, TypeNode] = {}
        sub_names = []
        for node in nodes:
            node_task_inputs: List[Union[TensorInput, TensorCompute]] = []
            for idx, tensor in enumerate(node.inputs):
                if tensor in input_tensors:
                    tensor_map[tensor] = node.task.params[idx]
                    tensor_type_map[tensor] = node.task.params_type[idx]
                node_task_inputs.append(tensor_map[tensor])
            for idx, out_tensor in enumerate(node.outputs):
                assert len(node.task.params[:-1]) == len(node.inputs) and all(isinstance(v, TensorInput) for v in node.task.params[:-1])
                node_task_output = rewrite(node.task.compute, rewrite_map={
                    task_tensor_input: tensor_map[tensor] for task_tensor_input, tensor in zip(node.task.params[:-1], node.inputs)
                })
                tensor_map[out_tensor] = node_task_output
                tensor_type_map[out_tensor] = node.task.type_of_param(node.task.compute)
            sub_names.append(node.task.name)
        task_inputs = [tensor_map[tensor] for tensor in input_tensors]
        task_outputs = [tensor_map[tensor] for tensor in output_tensors]
        task_outputs[0] = functors.inline_compute(task_outputs[0])
        task = Task(
            name="_".join(sub_names),
            computation=task_outputs[0],
            params=task_inputs + task_outputs,
            params_type=[tensor_type_map[tensor] for tensor in input_tensors + output_tensors],
            worker=nodes[0].task.worker
        )
        implementer = self.resolve_task(task)
        if implementer is None:
            return None
        if implementer.priority() < max(self.resolve_task(node.task).priority() for node in nodes):
            # fuse will degrade the implementer priority
            return None
        # name and attributes
        names = []
        attributes = {}
        for node in nodes:
            names.append(trim_fuse_head(node.name))
            attributes.update(node.attributes)
        return Operator(
            name='Fused' + "".join(names),
            task=task,
            inputs=input_tensors,
            outputs=output_tensors,
            **attributes
        )

    def resolve_task(self, task: Task) -> Optional[Implementer]:
        if task in self.task_resolve_cache:
            return self.task_resolve_cache[task]
        else:
            impl = hidet.implement.resolve_task(task)
            self.task_resolve_cache[task] = impl
            return impl


def fuse_elementwise_pass() -> GraphPass:
    return FuseElementwisePass()
