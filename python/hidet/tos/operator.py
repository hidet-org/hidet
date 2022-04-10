from typing import List, Optional, Dict, Any, Iterable, Tuple
from collections import OrderedDict, defaultdict
from hidet.ir.task import Task
from hidet.runtime import CompiledFunction
from hidet.driver import build_task
from hidet.tos.tensor import empty, Tensor, symbol
from hidet import utils


def trim_op_ending(name: str):
    return name[:-2] if name.endswith('Op') else name


class Operator:
    imperative_mode = 1
    lazy_mode = 2

    current_mode = imperative_mode
    current_opt_level = 0
    current_space_level = 0

    task_cache: Dict[Tuple[int, int], Dict[str, CompiledFunction]] = defaultdict(dict)

    def __init__(
            self,
            inputs: List[Tensor],
            task: Task,
            outputs: Optional[List[Tensor]] = None,
            name: Optional[str] = None,
            **kwargs):
        self.inputs: List[Tensor] = inputs
        self.task: Task = task
        self.attributes: Dict[str, Any] = kwargs
        self.outputs: Optional[List[Tensor]] = outputs
        name = name if name else trim_op_ending(self.__class__.__name__)
        self.name = name

        assert all(isinstance(v, Tensor) for v in inputs)

        # cache
        self.task_func: Optional[CompiledFunction] = None

    def __str__(self):
        arguments = ['{}: {}{}'.format(i, t.dtype, t.shape) for i, t in enumerate(self.inputs)]
        attributes = ['{}={}'.format(name, str(value)) for name, value in self.attributes.items()]
        return '{}({})'.format(self.name, ', '.join(arguments + attributes))

    def __dir__(self) -> Iterable[str]:
        return ['task', 'inputs', 'outputs', 'attributes', 'name'] + list(self.attributes)

    def __getattr__(self, item):
        if item in self.attributes:
            return self.attributes[item]
        else:
            raise AttributeError(item)

    def run(self) -> List[Tensor]:
        if self.current_mode == self.imperative_mode:
            return self.imperative_run(self.inputs)
        elif self.current_mode == self.lazy_mode:
            self.outputs = self.lazy_run()
            return self.outputs
        else:
            raise NotImplementedError('coming soon')

    def get_output(self, idx: int):
        if self.outputs is None:
            outputs = self.run()
        else:
            outputs = self.outputs
        return outputs[idx]

    def imperative_run(self, inputs: Optional[List[Tensor]] = None) -> List[Tensor]:
        if self.task_func is None:
            task_string = str(self.task)
            level = (self.current_space_level, self.current_opt_level)
            if task_string in self.task_cache[level]:
                self.task_func = self.task_cache[level][task_string]
            else:
                self.task_func = build_task(self.task, space_level=self.current_space_level, opt_level=self.current_opt_level, use_cache=True)
                self.task_cache[level][task_string] = self.task_func
        output_type = self.task.type_of_param(self.task.compute)
        outputs = [empty(shape=[int(v) for v in output_type.shape], dtype=output_type.scalar_type.name, layout=output_type.layout)]
        self.task_func(*inputs, *outputs)
        return outputs

    def lazy_run(self) -> List[Tensor]:
        output_type = self.task.type_of_param(self.task.compute)
        return [Tensor(shape=[int(v) for v in output_type.shape], dtype=output_type.scalar_type.name, device='cuda', storage=None, layout=output_type.layout, trace=(self, 0))]

    def clone(self, *new_inputs: Tensor):
        cls = self.__class__
        new_op = cls.__new__(cls)
        new_op.name = self.name
        new_op.inputs = list(new_inputs)
        new_op.task = self.task
        new_op.attributes = self.attributes
        new_op.outputs = new_op.lazy_run()
        new_op.task_func = None
        return new_op


def imperative_mode():
    Operator.current_mode = Operator.imperative_mode


def lazy_mode():
    Operator.current_mode = Operator.lazy_mode


def opt_level(level=0):
    Operator.current_opt_level = level


def space_level(level=0):
    Operator.current_space_level = level
