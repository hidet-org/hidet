from __future__ import annotations
from typing import Any, Deque
import copy
import os
import pickle
from collections import deque
from typing import Dict, List, Union, Optional, Sequence, Type, Tuple, Callable, TypeVar
from hidet.ir.node import Node
from hidet.ir.expr import Expr, Var, TensorElement, var
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

# todo: rename Prologue.indices and Epilogue.indices to axes


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
            return InverseMap.from_lambda(lambda *args: obj(*args))

    @staticmethod
    def from_lambda(func, num_args=None) -> InverseMap:
        num_args = num_args if num_args is not None else func.__code__.co_argcount
        axes = [var('v') for v in range(num_args)]
        indices = list(func(*axes))
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
            raise ValueError('Can not concat InverseMap a and b, '
                             'where a has {} indices and b has {} axes'.format(len(lhs.indices), len(rhs.axes)))
        rmap = {a: b for a, b in zip(rhs.axes, lhs.indices)}
        indices = [rewrite(index_expr, rmap) for index_expr in rhs.indices]
        return InverseMap(lhs.axes, indices)


class Prologue(Node):
    def __init__(self, extra_inputs, indices, value, inverse_map=None):
        if inverse_map is None:
            inverse_map = {}
        self.extra_inputs: List[TensorNode] = extra_inputs
        self.indices: List[Var] = indices
        self.value: Expr = value
        self.inverse_map: Dict[TensorNode, InverseMap] = {a: InverseMap.from_obj(b) for a, b in inverse_map.items()}
        self.bindings: Dict[TensorNode, TensorNode] = {}


class Epilogue(Node):
    def __init__(self, extra_inputs, indices, orig_value, value, out_indices, orig_tensor, out_tensor, bindings):
        self.extra_inputs: List[TensorNode] = extra_inputs

        self.indices: List[Var] = indices
        self.orig_value: Var = orig_value
        self.value: Expr = value
        self.out_indices: List[Expr] = out_indices

        self.orig_tensor: TensorNode = orig_tensor
        self.out_tensor: TensorNode = out_tensor

        self.bindings: Dict[TensorNode, TensorNode] = bindings


class TaskGraph(Node):
    def __init__(
            self,
            anchor: Task,
            nodes: Sequence[Task],
            consume: Dict[TensorNode, TensorNode],
            input_tensors: Sequence[TensorNode],
            output_tensors: Sequence[TensorNode]
    ):
        self.anchor: Task = anchor
        self.nodes: List[Task] = list(nodes)
        self.input_tensors: List[TensorNode] = list(input_tensors)
        self.output_tensors: List[TensorNode] = list(output_tensors)
        self.consume: Dict[TensorNode, TensorNode] = consume

    @staticmethod
    def from_task(task: Task) -> TaskGraph:
        return TaskGraph(
            anchor=task,
            nodes=[task],
            consume={},
            input_tensors=task.inputs,
            output_tensors=task.outputs
        )

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
        return Task(
            name=self.anchor.name,
            inputs=graph_input_tensors,
            outputs=graph_output_tensors
        )


class TaskContext:
    """Scheduling task context.
    """
    contexts = []

    def __init__(self, space_level: int = 0, warmup: int = 3, number: int = 10, repeat: int = 3, resolve_out_dir: str = None):
        """Create a task context.

        Parameters
        ----------
        space_level: int
            The search space level. Can be 0, 1, or 2. Larger space level indicates larger search space.
        warmup: int
            The number of warmups for tuning.
        number: int
            The number of runs per repeat.
        repeat: int
            The number of repeats.
        resolve_out_dir: Optional[str]
            The output directory for the intermediate output for tuning.
        """
        self.space_level = space_level
        self.warmup: int = warmup
        self.number: int = number
        self.repeat: int = repeat
        self.resolve_out_dir = resolve_out_dir

    def __enter__(self):
        self.contexts.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.contexts.pop()

    @staticmethod
    def current() -> TaskContext:
        return TaskContext.contexts[-1]


TaskContext.contexts.append(TaskContext())  # fallback context


class Task(Node):
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

    @property
    def parameters(self) -> List[TensorNode]:
        return self.task_graph.input_tensors + self.task_graph.output_tensors

    def implement(self, target: Union[Target, str]) -> IRModule:
        from hidet.graph.ops.schedules.cuda.auto_scheduler import CudaAutoScheduler
        from hidet.graph.ops.schedules.cpu.auto_scheduler import CpuAutoScheduler
        if isinstance(target, str):
            target = Target.from_string(target)
        if target.name == 'cuda':
            ret = self.implement_cuda()
            if ret is NotImplemented:
                auto_scheduler = CudaAutoScheduler()
                ret = auto_scheduler.schedule_task(self, 'cuda')
        elif target.name == 'cpu':
            ret = self.implement_cpu()
            if ret is NotImplemented:
                auto_scheduler = CpuAutoScheduler()
                ret = auto_scheduler.schedule_task(self, 'cpu')
        else:
            raise ValueError()
        if not isinstance(ret, IRModule):
            raise AssertionError('The task implement function should return an IRModule, but got a {}.'.format(type(ret)))
        return ret

    def implement_cuda(self) -> IRModule:
        return NotImplemented

    def implement_cpu(self) -> IRModule:
        return NotImplemented

    def allow_prologue(self) -> True:
        return True

    def allow_epilogue(self) -> True:
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

