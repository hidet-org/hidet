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
from __future__ import annotations
from typing import Sequence, List, Dict, Union, Tuple, Optional
from collections import defaultdict
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
    def __init__(self, fused_graph: FlowGraph, anchor: int):
        inputs, outputs = self._computation(fused_graph)
        super().__init__(
            name='fused',
            inputs=inputs,
            outputs=outputs,
            attributes={'fused_ops': self._fused_name(fused_graph), 'anchor': fused_graph.nodes[anchor].task.name},
        )
        self.fused_graph: FlowGraph = fused_graph
        self.anchor: int = anchor

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return False

    @staticmethod
    def _fused_name(flow_graph: FlowGraph):
        names = []
        for op in flow_graph.nodes:
            names.append(op.task.name)
        return ' '.join(names)

    def _computation(self, fused_graph: FlowGraph) -> Tuple[List[TensorInput], List[TensorNode]]:
        """
        Get the computation definition of the fused subgraph.

        Parameters
        ----------
        fused_graph: FlowGraph
            The fused subgraph

        Returns
        -------
        inputs, outputs: List[TensorInput], List[TensorNode]
            The inputs and outputs of the fused subgraph in the compute IR defined in hidet.ir.compute module.
        """
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
        from hidet.graph.ops.definitions.utils.tune import tune

        if isinstance(target, str):
            target = Target.from_string(target)

        anchor_op = self.fused_graph.nodes[self.anchor]

        if target.name == 'cpu':
            anchor_modules: Union[NotImplemented, IRModule] = anchor_op.task.implement_cpu(working_dir)
            if anchor_modules is NotImplemented:
                auto_scheduler = CpuAutoScheduler()
                return auto_scheduler.schedule_task(self, 'cpu')
        elif target.name == 'cuda':
            anchor_modules: Union[NotImplemented, IRModule] = anchor_op.task.implement_cuda(working_dir)
            if anchor_modules is NotImplemented:
                auto_scheduler = CudaAutoScheduler()
                return auto_scheduler.schedule_task(self, 'cuda')
        else:
            raise ValueError('unsupported target: {}'.format(target))

        if isinstance(anchor_modules, IRModule):
            anchor_modules = [anchor_modules]

        fused_modules: List[IRModule] = [apply_prologue_epilogue(m, self, working_dir) for m in anchor_modules]
        for fused_module, anchor_module in zip(fused_modules, anchor_modules):
            if hasattr(anchor_module, '_tuning_kwargs'):
                setattr(fused_module, '_tuning_kwargs', getattr(anchor_module, '_tuning_kwargs'))

        if len(fused_modules) == 1:
            return fused_modules[0]

        return tune(fused_modules, dummy_inputs=self.dummy_arguments(target.name), working_dir=working_dir)


class FusedOperator(Operator):
    def __init__(self, *inputs: Tensor, fused_graph: FlowGraph, anchor: int):
        task = FusedTask(fused_graph, anchor)
        super().__init__(inputs=list(inputs), attributes={'fused_graph': fused_graph, 'anchor': anchor}, task=task)
        self.name = f'Fused{fused_graph.nodes[anchor].name}'
        self._check(inputs, fused_graph, anchor)

    @staticmethod
    def _check(inputs: Sequence[Tensor], fused_graph: FlowGraph, anchor_idx: int):
        # check the input shapes and dtypes match
        if len(inputs) != len(fused_graph.inputs):
            raise ValueError('number of inputs mismatch')
        for idx, (a, b) in enumerate(zip(inputs, fused_graph.inputs)):
            if not same_list(a.shape, b.shape):
                raise ValueError('Arg {} shape mismatch: {} vs {}'.format(idx, a.shape, b.shape))
            if a.dtype != b.dtype:
                raise ValueError('Arg {} dtype mismatch: {} vs {}'.format(idx, a.dtype, b.dtype))

        # check the anchor operator's prologue & epilogue requirement is satisfied
        nodes: List[Operator] = fused_graph.nodes
        anchor: Operator = fused_graph.nodes[anchor_idx]
        if anchor not in nodes:
            raise ValueError('anchor operator not in the subgraph')
        if any(x not in fused_graph.inputs for x in anchor.inputs) and not anchor.task.allow_prologue():
            raise ValueError('found prologue but it is not allowed for anchor operator')
        if any(x not in fused_graph.outputs for x in anchor.outputs) and not anchor.task.allow_epilogue():
            raise ValueError('found epilogue but it is not allowed for anchor operator')


def _anchor_op_idx(fused_graph: FlowGraph) -> int:
    non_injective_ops = [i for i, op in enumerate(fused_graph.nodes) if not op.task.is_injective()]
    if len(non_injective_ops) == 0:
        return len(fused_graph.nodes) - 1
    elif len(non_injective_ops) == 1:
        return non_injective_ops[0]
    else:
        raise ValueError('multiple non-injective operators found in the fused graph')


def fused_operator(
    *inputs: Tensor, fused_graph: FlowGraph, anchor: Optional[int] = None
) -> Union[Tensor, List[Tensor]]:
    if anchor is None:
        anchor = _anchor_op_idx(fused_graph)
    op = FusedOperator(*inputs, fused_graph=fused_graph, anchor=anchor)
    outputs = []
    for i in range(len(fused_graph.outputs)):
        outputs.append(op.get_output(i))
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs
