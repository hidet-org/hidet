from __future__ import annotations
from typing import Sequence, List, Dict, Union, Tuple, Optional
from collections import defaultdict
from hidet.ir.node import Node
from hidet.ir.task import Task, Target
from hidet.ir.func import IRModule
from hidet.ir.compute import TensorNode, TensorInput
from hidet.graph.tensor import Tensor
from hidet.graph.operator import Operator
from hidet.graph.ir import FlowGraph
from hidet.graph.ops.definitions.utils import input_like
from hidet.ir.tools import rewrite
from hidet.utils import same_list, index_of


class FusedTask(Task):
    def __init__(self, fused_graph: FlowGraph, anchor: Operator):
        inputs, outputs = self._computation(fused_graph)
        super().__init__(
            name='fused', inputs=inputs, outputs=outputs, attributes={'fused_ops': self._fused_name(fused_graph)}
        )
        self.fused_graph: FlowGraph = fused_graph
        self.anchor: Operator = anchor

    def _fused_name(self, flow_graph: FlowGraph):
        names = []
        for op in flow_graph.nodes:
            names.append(op.task.name)
        return ' '.join(names)

    def _computation(self, fused_graph: FlowGraph) -> Tuple[List[TensorInput], List[TensorNode]]:
        inputs: List[TensorInput] = []
        consumer: Dict[Tensor, List[Operator]] = defaultdict(list)
        tensor_map: Dict[Tensor, TensorNode] = {}

        for op in fused_graph.nodes:
            if isinstance(op, FusedOperator):
                raise NotImplementedError('nested fusion is not supported')
            for x in op.inputs:
                consumer[x].append(op)

        for x in fused_graph.inputs:
            if len(consumer[x]) == 0:
                inputs.append(input_like(x, 'unused'))
            else:
                op = consumer[x][0]  # pick the first consumer to name the input
                task: Task = op.task
                name: str = task.inputs[index_of(x, consumer[x][0].inputs)].name
                inputs.append(input_like(x, name))
            tensor_map[x] = inputs[-1]

        for op in fused_graph.nodes:
            task: Task = op.task
            remap: Dict[TensorNode, TensorNode] = {a: tensor_map[b] for a, b in zip(op.task.inputs, op.inputs)}
            op_outputs: List[TensorNode] = [rewrite(x, remap) for x in task.outputs]
            tensor_map.update({a: b for a, b in zip(op.outputs, op_outputs)})

        outputs: List[TensorNode] = [tensor_map[x] for x in fused_graph.outputs]
        return inputs, outputs

    def implement(self, target: Union[Target, str], working_dir: str) -> IRModule:
        from hidet.graph.ops.schedules.cuda.auto_scheduler import CudaAutoScheduler
        from hidet.graph.ops.schedules.cpu.auto_scheduler import CpuAutoScheduler
        from .apply_prologue_epilogue import apply_prologue_epilogue

        if isinstance(target, str):
            target = Target.from_string(target)

        if target.name == 'cpu':
            anchor_module: Union[NotImplemented, IRModule] = self.anchor.task.implement_cpu(working_dir)
            if anchor_module is NotImplemented:
                auto_scheduler = CpuAutoScheduler()
                return auto_scheduler.schedule_task(self, 'cuda')
        elif target.name == 'cuda':
            anchor_module: Union[NotImplemented, IRModule] = self.anchor.task.implement_cuda(working_dir)
            if anchor_module is NotImplemented:
                auto_scheduler = CudaAutoScheduler()
                return auto_scheduler.schedule_task(self, 'cuda')
        else:
            raise ValueError('unsupported target: {}'.format(target))

        # we have scheduled the anchor operator, now we need to fuse the rest of the graph
        fused_module: IRModule = apply_prologue_epilogue(anchor_module, self)
        return fused_module


class FusedOperator(Operator):
    def __init__(self, *inputs: Tensor, fused_graph: FlowGraph, anchor: Operator):
        task = FusedTask(fused_graph, anchor)
        super().__init__(
            inputs=list(inputs),
            task=task,
            name=f'Fused{anchor.name}',
            attributes={'fused_graph': fused_graph, 'anchor': anchor},
        )

        self._check(inputs, fused_graph)

    def _check(self, inputs: Sequence[Tensor], fused_graph: FlowGraph):
        # check the input shapes and dtypes match
        if len(inputs) != len(fused_graph.inputs):
            raise ValueError('number of inputs mismatch')
        for idx, (a, b) in enumerate(zip(inputs, fused_graph.inputs)):
            if not same_list(a.shape, b.shape):
                raise ValueError('Arg {} shape mismatch: {} vs {}'.format(idx, a.shape, b.shape))
            if a.dtype != b.dtype:
                raise ValueError('Arg {} dtype mismatch: {} vs {}'.format(idx, a.dtype, b.dtype))

        # check the subgraph is valid
        # todo


def fused_operator(*inputs: Tensor, fused_graph: FlowGraph, anchor: Operator) -> Union[Tensor, List[Tensor]]:
    op = FusedOperator(*inputs, fused_graph=fused_graph, anchor=anchor)
    outputs = []
    for i in range(len(fused_graph.outputs)):
        outputs.append(op.get_output(i))
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs
