from __future__ import annotations
from typing import Any
import copy
import os
import pickle
from typing import Dict, List, Union, Optional, Sequence, Type, Tuple, Callable, TypeVar
from hidet.ir.node import Node
from hidet.ir.expr import Expr, Var, TensorElement, var
from hidet.ir.func import IRModule
from hidet.ir.dialects.compute import TensorNode, ScalarNode, GridCompute


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


class Prologue(Node):
    def __init__(self, extra_inputs, indices, value):
        self.extra_inputs: List[TensorNode] = extra_inputs
        self.indices: List[Var] = indices
        self.value: Expr = value


class Epilogue(Node):
    def __init__(self, extra_inputs, indices, orig_value, value, out_indices, out_tensor):
        self.extra_inputs: List[TensorNode] = extra_inputs
        self.indices: List[Var] = indices
        self.orig_value: Var = orig_value
        self.value: Expr = value
        self.out_indices: List[Expr] = out_indices
        self.out_tensor: TensorNode = out_tensor


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


class InverseMap:
    def __init__(self, axes: List[Var], indices: List[Expr]):
        from hidet.ir.functors import simplify
        self.axes: List[Var] = axes
        self.indices: List[Expr] = [simplify(e) for e in indices]

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


class Task(Node):
    def __init__(self, name, inputs, outputs, prologues=None, epilogues=None, parameters=None, inverse_map=None, attributes: Optional[Dict] = None):
        """
        A Task is a computation definition.

        param_inputs ===========>  task_inputs  ===================> task_outputs ===========> param_outputs
             |        prologues                  task computations                 epilogues
             |           ^                                                             ^
             v           |                                                             |
             +-----------+-------------------------------------------------------------+

        Constraints:
            1. Each task input can have zero or one prologue.
            2. Each task output can have zero or one epilogue.
            3. Prologue and epilogue can only have extra inputs from param inputs.
            4. When a task input has prologue, it should not appear in param input.
            5. When a task output has epilogue, it should not appear in param output.


        Parameters
        ----------
        name: str
            The name of the task. Can only contain a-z, A-Z, underscore, and digits.
        inputs: List[TensorNode]
            The inputs of the task computation.
        outputs: List[TensorNode]
            The outputs of the task computation.
        prologues: Dict[TensorNode, Prologue]
            The prologues.
        epilogues: Dict[TensorNode, Epilogue]
            The epilogues.
        parameters: List[TensorNode]
            The list of parameters in the final kernel.
        inverse_map: Dict[TensorNode, Union[InverseMap, Callable[[Any], Any]]]
            If the mapping of input axes to output axes are invertible, then inverse_map contains
            the inverse map. It is used to convert a task to epilogue of previous task.
        """
        self.attributes: Dict[str, Union[str, float, int, bool]] = attributes if attributes is not None else {}
        self.name = name
        self.inputs: List[TensorNode] = inputs
        self.outputs: List[TensorNode] = outputs
        self.prologues: Dict[TensorNode, Prologue] = prologues if prologues else {}
        self.epilogues: Dict[TensorNode, Epilogue] = epilogues if epilogues else {}
        self.parameters: List[TensorNode] = parameters if parameters else inputs + outputs

        inverse_map = inverse_map if inverse_map else {}
        if not isinstance(inverse_map, dict):
            raise ValueError('inverse_map should be a dict')
        self.inverse_map: Dict[TensorNode, InverseMap] = {
            a: (b if isinstance(b, InverseMap) else InverseMap.from_lambda(b)) for a, b in inverse_map.items()
        }

    def implement(self, target: Union[Target, str]) -> IRModule:
        from hidet.graph.ops.schedules import generic_cuda_schedule, generic_cpu_schedule
        if isinstance(target, str):
            target = Target.from_string(target)
        if target.name == 'cuda':
            ret = self.implement_cuda()
            if ret is NotImplemented:
                ret = generic_cuda_schedule(self)
        elif target.name == 'cpu':
            ret = self.implement_cpu()
            if ret is NotImplemented:
                ret = generic_cpu_schedule(self)
        else:
            raise ValueError()
        if not isinstance(ret, IRModule):
            raise AssertionError('The task implement function should return an IRModule, but got a {}.'.format(type(ret)))
        return ret

    def implement_cuda(self) -> IRModule:
        return NotImplemented

    def implement_cpu(self) -> IRModule:
        return NotImplemented

    def allow_prologue(self, only_elementwise=False) -> True:
        return True

    def allow_epilogue(self, only_elementwise=False) -> True:
        return True

    def fast_implement(self, space_level: int) -> bool:
        """
        Whether the function can be implemented through a single thread.

        Note:
        When we implement a task, we might try different schedules by launching multiple compilation processes.
        This prevents us from implementing such kind of task with other tasks in the model. Thus, we usually
        parallelize the implementing of tasks that only use a single thread during their implementing. Then
        implement those require multiple threads one by one.

        Parameters
        ----------
        space_level: int
            The space level to explore during implementing.

        Returns
        -------
        ret: bool
            True if this task can be implemented through a single cpu thread.
        """
        if space_level == 0:
            return True
        else:
            if 'implement_cuda' not in self.__class__.__dict__:
                return True
            else:
                return False

    def copy(self: Task) -> Task:
        cls = type(self)
        task = object.__new__(cls)
        task.name = self.name
        task.inputs = self.inputs.copy()
        task.outputs = self.outputs.copy()
        task.prologues = self.prologues.copy()
        task.epilogues = self.epilogues.copy()
        task.parameters = self.parameters.copy()
        task.inverse_map = self.inverse_map.copy()
        for name in self.__dict__:
            if name not in task.__dict__:
                task.__dict__[name] = copy.copy(self.__dict__[name])
        return task

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

