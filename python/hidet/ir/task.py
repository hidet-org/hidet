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
# pylint: disable=import-outside-toplevel
from __future__ import annotations
from typing import Any, Dict, List, Union, Optional, Sequence, Callable
import os
import pickle
from hidet.ir.node import Node
from hidet.ir.expr import Expr, Var, var
from hidet.ir.func import IRModule
from hidet.ir.compute import TensorNode, ScalarNode, GridCompute


class Target:
    _supported_targets = ['cuda', 'cpu']

    def __init__(self, name: str, attrs: List[str]):
        if name not in self._supported_targets:
            raise ValueError('Does not support target {}, candidates {}.'.format(name, self._supported_targets))
        self.name = name
        self.attrs = attrs

    @staticmethod
    def from_string(target_string: str) -> Target:
        items = target_string.split()
        name, attrs = items[0], items[1:]
        return Target(name, attrs)


class InverseMap(Node):
    def __init__(self, axes: List[Var], indices: List[Expr]):
        from hidet.ir.functors import simplify

        self.axes: List[Var] = axes
        self.indices: List[Expr] = [simplify(e) for e in indices]

    @staticmethod
    def from_obj(obj: Union[InverseMap, Callable[[Any], Any]]):
        if isinstance(obj, InverseMap):
            return obj
        else:
            return InverseMap.from_lambda(lambda *args: obj(*args))  # pylint: disable=unnecessary-lambda

    @staticmethod
    def from_lambda(func, num_args=None) -> InverseMap:
        from hidet.ir.utils import as_expr

        num_args = num_args if num_args is not None else func.__code__.co_argcount
        axes = [var('v') for v in range(num_args)]
        indices = [as_expr(index_expr) for index_expr in func(*axes)]
        return InverseMap(axes, indices)

    @staticmethod
    def identity(num_args: int) -> InverseMap:
        return InverseMap.from_lambda(lambda *indices: list(indices), num_args=num_args)

    def __add__(self, other) -> InverseMap:
        from hidet.ir.functors import rewrite

        if not isinstance(other, InverseMap):
            raise ValueError('Can not concat InverseMap with {}'.format(type(other)))
        lhs, rhs = self, other
        if len(lhs.indices) != len(rhs.axes):
            raise ValueError(
                'Can not concat InverseMap a and b, '
                'where a has {} indices and b has {} axes'.format(len(lhs.indices), len(rhs.axes))
            )
        rmap = dict(zip(rhs.axes, lhs.indices))
        indices = [rewrite(index_expr, rmap) for index_expr in rhs.indices]
        return InverseMap(lhs.axes, indices)


class TaskGraph(Node):
    def __init__(
        self,
        anchor: Task,
        nodes: Sequence[Task],
        consume: Dict[TensorNode, TensorNode],
        input_tensors: Sequence[TensorNode],
        output_tensors: Sequence[TensorNode],
    ):
        self.anchor: Task = anchor
        self.nodes: List[Task] = list(nodes)
        self.input_tensors: List[TensorNode] = list(input_tensors)
        self.output_tensors: List[TensorNode] = list(output_tensors)
        self.consume: Dict[TensorNode, TensorNode] = consume

    @staticmethod
    def from_task(task: Task) -> TaskGraph:
        return TaskGraph(anchor=task, nodes=[task], consume={}, input_tensors=task.inputs, output_tensors=task.outputs)

    def absorb(self) -> Task:
        from hidet.ir.functors import rewrite

        graph_input_tensors: List[TensorNode] = self.input_tensors
        update_map: Dict[TensorNode, TensorNode] = {a: a for a in graph_input_tensors}
        for task in self.nodes:
            remap: Dict[TensorNode, TensorNode] = {}
            for task_input_tensor in task.inputs:
                if task_input_tensor not in self.consume:
                    assert task_input_tensor in graph_input_tensors, 'must be graph input tensor'
                    # do not need to rewrite, skip
                else:
                    remap[task_input_tensor] = update_map[self.consume[task_input_tensor]]
            for task_output_tensor in task.outputs:
                update_map[task_output_tensor] = rewrite(task_output_tensor, remap)
                # original_output_tensor = task_output_tensor
                # updated_output = rewrite(task_output_tensor, remap)
                # update_map[original_output_tensor] = updated_output
        graph_output_tensors: List[TensorNode] = [update_map[tensor] for tensor in self.output_tensors]
        return Task(name=self.anchor.name, inputs=graph_input_tensors, outputs=graph_output_tensors)


class Task(Node):
    """
    A task defines the operator computation.
    """

    def __init__(self, name, inputs, outputs, inverse_map=None, attributes=None):
        if inverse_map is None:
            inverse_map = {}
        if attributes is None:
            attributes = {}
        self.name: str = name
        self.inputs: List[TensorNode] = inputs
        self.outputs: List[TensorNode] = outputs
        self.attributes: Dict[str, Union[str, float, int, bool]] = attributes
        self.inverse_map: Dict[TensorNode, InverseMap] = {a: InverseMap.from_obj(b) for a, b in inverse_map.items()}
        self.task_graph: Optional[TaskGraph] = TaskGraph.from_task(self)

        # sanity check
        for tn, im in self.inverse_map.items():
            if len(im.axes) != tn.ndim:
                raise ValueError(
                    'InverseMap for tensor {} has {} input axes, but input tensor has {} axes'.format(
                        tn.name, len(im.axes), tn.ndim
                    )
                )
            if len(im.indices) != self.outputs[0].ndim:
                raise ValueError(
                    'InverseMap for tensor {} has {} output indices, but output tensor has {} axes'.format(
                        tn.name, len(im.indices), self.outputs[0].ndim
                    )
                )

    def signature(self) -> str:
        params = []
        for tensor in self.inputs:
            name = tensor.name
            dtype = tensor.ttype.dtype.name
            shape = tensor.const_shape()
            params.append('{}={}{}'.format(name, dtype, shape))
        for name, value in self.attributes.items():
            params.append('{}={}'.format(name, value))
        param_doc = ', '.join(params)
        fuse_doc = ''
        if len(self.task_graph.nodes) > 1:
            fuse_doc = ' ({} fused)'.format(len(self.task_graph.nodes) - 1)
        return ''.join([self.name, '(', param_doc, ')', fuse_doc])

    @property
    def parameters(self) -> List[TensorNode]:
        return self.task_graph.input_tensors + self.task_graph.output_tensors

    def implement(self, target: Union[Target, str], workding_dir: str) -> IRModule:
        from hidet.graph.ops.schedules.cuda.auto_scheduler import CudaAutoScheduler
        from hidet.graph.ops.schedules.cpu.auto_scheduler import CpuAutoScheduler

        if isinstance(target, str):
            target = Target.from_string(target)
        if target.name == 'cuda':
            ret = self.implement_cuda(workding_dir)
            if ret is NotImplemented:
                auto_scheduler = CudaAutoScheduler()
                ret = auto_scheduler.schedule_task(self, 'cuda')
        elif target.name == 'cpu':
            ret = self.implement_cpu(workding_dir)
            if ret is NotImplemented:
                auto_scheduler = CpuAutoScheduler()
                ret = auto_scheduler.schedule_task(self, 'cpu')
        else:
            raise ValueError()
        if not isinstance(ret, IRModule):
            raise AssertionError(f'The task implement function should return an IRModule, but got a {type(ret)}.')
        return ret

    def implement_cuda(self, workding_dir: str) -> IRModule:
        return NotImplemented

    def implement_cpu(self, workding_dir: str) -> IRModule:
        return NotImplemented

    def allow_prologue(self) -> bool:
        return True

    def allow_epilogue(self) -> bool:
        return True

    def is_injective_task(self) -> bool:
        from hidet.ir.functors import collect

        if len(self.outputs) != 1 or not isinstance(self.outputs[0].tensor_compute, GridCompute):
            return False

        scalar_nodes: List[ScalarNode] = collect(self.outputs, ScalarNode, stop_when_found=False)
        for scalar_node in scalar_nodes:
            if scalar_node.scalar_compute is not None:
                return False

        tensor_nodes: List[TensorNode] = collect(self.outputs, TensorNode, stop_when_found=False)
        for tensor_node in tensor_nodes:
            tensor_compute = tensor_node.tensor_compute
            if tensor_compute is not None and not isinstance(tensor_compute, GridCompute):
                return False

        return True

    def save(self, fname: str):
        dirname = os.path.dirname(fname)
        os.makedirs(dirname, exist_ok=True)
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname: str) -> Task:
        with open(fname, 'rb') as f:
            return pickle.load(f)


def is_injective_task(task: Task) -> bool:
    """
    Check whether a task is an injective task. A task is injective if and only if there is no reduce compute in
    the task.

    Parameters
    ----------
    task: Task
        The task to check.

    Returns
    -------
    ret: bool
        Whether the task is injective.
    """
    from hidet.ir.functors import collect

    scalar_nodes: List[ScalarNode] = collect(task.outputs, ScalarNode, stop_when_found=False)
    return all(sn.scalar_compute is None for sn in scalar_nodes)


def is_unary_injective_task(task: Task) -> bool:
    return len(task.inputs) == 1 and len(task.outputs) == 1 and is_injective_task(task)


def is_elementwise_task(task: Task) -> bool:
    """
    Check whether a task is an elementwise task. A task is elementwise if and only if it is a unary injective and
    invertible.

    Parameters
    ----------
    task: Task
        The task to check.

    Returns
    -------
    ret: bool
        Whether the task is elementwise.
    """
    return is_unary_injective_task(task) and len(task.inverse_map) > 0


def save_task(task: Task, fname: str):
    task.save(fname)


def load_task(fname: str) -> Task:
    return Task.load(fname)
