from __future__ import annotations
import copy
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
    contexts = []

    def __init__(self, space_level: int = 0, resolve_out_dir: str = None):
        self.space_level = space_level
        self.resolve_out_dir = resolve_out_dir

    def __enter__(self):
        self.contexts.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.contexts.pop()

    @staticmethod
    def current() -> TaskContext:
        return TaskContext.contexts[-1]


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


TaskType = TypeVar('TaskType')


class Task(Node):
    def __init__(self, name, inputs, outputs, prologues=None, epilogues=None, parameters=None, inverse_map=None):
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
        self.name = name
        self.inputs: List[TensorNode] = inputs
        self.outputs: List[TensorNode] = outputs
        self.prologues: Dict[TensorNode, Prologue] = prologues if prologues else {}
        self.epilogues: Dict[TensorNode, Epilogue] = epilogues if epilogues else {}
        self.parameters: List[TensorNode] = parameters if parameters else inputs + outputs

        inverse_map = inverse_map if inverse_map else {}
        self.inverse_map: Dict[TensorNode, InverseMap] = {
            a: (b if isinstance(b, InverseMap) else InverseMap.from_lambda(b)) for a, b in inverse_map.items()
        }

        sanity_check(self)

    def implement(self, target: Union[Target, str]) -> IRModule:
        if isinstance(target, str):
            target = Target.from_string(target)
        if target.name == 'cuda':
            ret = self.implement_cuda()
        elif target.name == 'cpu':
            ret = self.implement_cpu()
        else:
            raise ValueError()
        if not isinstance(ret, IRModule):
            raise AssertionError('The task implement function should return an IRModule, but got a {}.'.format(type(ret)))
        return ret

    def implement_cuda(self) -> IRModule:
        from hidet.tos.ops.schedules import generic_cuda_schedule
        return generic_cuda_schedule(self)

    def implement_cpu(self) -> IRModule:
        from hidet.tos.ops.schedules import generic_cpu_schedule
        return generic_cpu_schedule(self)

    def copy(self: TaskType) -> TaskType:
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

    def absorb(self) -> Task:
        """
        Get a task that absorbed the prologues and epilogues into the main body of the computation.
        This would make the specialized schedule inapplicable to the new task.

        Returns
        -------
        ret: Task
            The task absorbed the prologues and epilogues. Thus, its prologue and epilogues are empty.

        """
        from hidet.ir.functors import collect, rewrite
        if len(self.prologues) + len(self.epilogues) == 0:
            # current task have no prologue and epilogue. do nothing.
            return self
        num_outputs = len(self.outputs)
        num_inputs = len(self.parameters) - num_outputs
        global_inputs = self.parameters[:num_inputs]
        for input_node in global_inputs:
            if input_node.grid_compute:
                raise ValueError('The input node of task should not have compute definition.')

        # original tensor node to new tensor node.
        tensor_map: Dict[TensorNode, TensorNode] = {}

        # apply prologues
        for task_input in self.inputs:
            if task_input in self.prologues:
                prologue = self.prologues[task_input]
                tensor_map[task_input] = TensorNode(
                    task_input.name,
                    data_type=task_input.data_type,
                    grid_compute=GridCompute(
                        shape=task_input.grid_compute.shape,
                        axes=prologue.indices,
                        value=prologue.value
                    )
                )
            else:
                tensor_map[task_input] = task_input

        # apply task computations and epilogues
        for task_output in self.outputs:
            new_output = TensorNode(
                name=task_output.name,
                data_type=task_output.data_type,
                grid_compute=GridCompute(
                    shape=task_output.grid_compute.shape,
                    axes=task_output.grid_compute.axes,
                    value=rewrite(task_output.grid_compute.value, tensor_map)
                )
            )
            tensor_map[task_output] = new_output
            if task_output in self.epilogues:
                epilogue = self.epilogues[task_output]
                tensor_map[task_output] = TensorNode(
                    name=epilogue.out_tensor.name,
                    data_type=epilogue.out_tensor.data_type,
                    grid_compute=GridCompute(
                        shape=epilogue.out_tensor.grid_compute.shape,
                        axes=epilogue.out_tensor.grid_compute.axes,
                        value=rewrite(epilogue.out_tensor.grid_compute.value, tensor_map)
                    )
                )
        global_outputs = [tensor_map[task_output] for task_output in self.outputs]
        return Task(
            name=self.name,
            inputs=global_inputs,
            outputs=global_outputs,
        )


def is_elementwise(task: Task) -> bool:
    """
    A task is elementwise if and only if there is only no reduce compute in the task.

    Parameters
    ----------
    task: Task
        The task to check.

    Returns
    -------
    ret: bool
        Whether the task is elementwise.
    """
    from hidet.ir.functors import collect
    scalar_nodes: List[ScalarNode] = collect(task.outputs, ScalarNode, stop_when_found=False)
    return all(sn.reduce_compute is None for sn in scalar_nodes)


def is_unary_elementwise(task: Task) -> bool:
    return len(task.inputs) == 1 and len(task.outputs) == 1 and is_elementwise(task)


def sanity_check(task: Task):
    from hidet.ir.functors import collect
    # todo: check
    pass
