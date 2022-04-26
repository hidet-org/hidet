from typing import List, Optional, Dict, Any, Iterable, Tuple, Union
from collections import defaultdict

from hidet.ir.task import Task
from hidet.runtime import CompiledFunction
from hidet.driver import build_task
from hidet.tos.tensor import empty, Tensor


def trim_op_ending(name: str):
    return name[:-2] if name.endswith('Op') else name


NTask = 'hidet.tos.task.Task'


class Operator:
    _current_opt_level = 0
    _current_space_level = 0
    _use_cache = True

    _task_cache: Dict[Tuple[int, int], Dict[str, CompiledFunction]] = defaultdict(dict)

    def __init__(
            self,
            inputs: List[Tensor],
            task: Union[Task, NTask],
            outputs: Optional[List[Tensor]] = None,
            name: Optional[str] = None,
            **kwargs):
        self.inputs: List[Tensor] = inputs
        self.task: Union[Task, NTask] = task
        self.attributes: Dict[str, Any] = kwargs
        self.outputs: Optional[List[Tensor]] = outputs
        self.name = name if name else trim_op_ending(self.__class__.__name__)

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
        if all(t.storage is not None for t in self.inputs):
            return self.imperative_run(self.inputs)
        else:
            self.outputs = self.lazy_run()
            return self.outputs

    def get_output(self, idx: int) -> Tensor:
        if self.outputs is None:
            outputs = self.run()
        else:
            outputs = self.outputs
        return outputs[idx]

    # @hidet.utils.line_profile()
    def imperative_run(self, inputs: List[Tensor]) -> List[Tensor]:
        if self.task_func is None:
            task_string = str(self.task)
            level = (self._current_space_level, self._current_opt_level)
            if task_string in self._task_cache[level]:
                self.task_func = self._task_cache[level][task_string]
            else:
                self.task_func = build_task(self.task, space_level=self._current_space_level, opt_level=self._current_opt_level, use_cache=self._use_cache)
                self._task_cache[level][task_string] = self.task_func
        output_type = self.task.compute.data_type
        outputs = [empty(shape=[int(v) for v in output_type.shape], dtype=output_type.scalar_type.name, layout=output_type.layout)]
        self.task_func(*inputs, *outputs)
        return outputs

    def lazy_run(self) -> List[Tensor]:
        output_type = self.task.compute.data_type
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


def opt_level(level=0):
    Operator._current_opt_level = level


def space_level(level=0):
    Operator._current_space_level = level


def cache_operator(use_cache=True):
    Operator._use_cache = use_cache
