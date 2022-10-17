from typing import List, Optional, Dict, Any, Iterable, Tuple, Union
from collections import defaultdict, namedtuple

from hidet.ir.task import Task
from hidet.runtime.module import CompiledFunction
from hidet.graph.tensor import empty, empty_like, Tensor
from hidet.ffi.ffi import get_last_error, BackendException


def trim_op_ending(name: str):
    return name[:-2] if name.endswith('Op') else name


ProfileConfig = namedtuple('ProfileConfig', ['warmup', 'number', 'repeat'])

_profile_config = ProfileConfig(warmup=3, number=10, repeat=3)


class Operator:
    """An operator that takes tensor as input and output.
    """
    _current_space_level = 0
    _use_cache = True

    def __init__(
            self,
            inputs: List[Tensor],
            task: Optional[Task],
            outputs: Optional[List[Tensor]] = None,
            name: Optional[str] = None,
            attributes: Optional[Dict[str, Any]] = None):
        self.inputs: List[Tensor] = inputs
        self.task: Optional[Task] = task
        self.attrs: Dict[str, Any] = attributes if attributes is not None else {}
        self.outputs: Optional[List[Tensor]] = outputs
        self.name: str = name if name else trim_op_ending(self.__class__.__name__)

        # assert len(inputs) > 0 and all(inputs[i].device == inputs[0].device for i in range(len(inputs))), 'Expect input tensor on the same device'
        self.device: str = inputs[0].device

        assert all(isinstance(v, Tensor) for v in inputs)

        # cache
        self.task_func: Optional[CompiledFunction] = None

    def __str__(self):
        arguments = ['{}: {}{}'.format(i, t.dtype, t.shape) for i, t in enumerate(self.inputs)]
        attributes = ['{}={}'.format(name, str(value)) for name, value in self.attrs.items()]
        return '{}({})'.format(self.name, ', '.join(arguments + attributes))

    # def __dir__(self) -> Iterable[str]:
    #     return ['task', 'inputs', 'outputs', 'attributes', 'name', 'device'] + list(self.attrs)

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

    def imperative_run(self, inputs: List[Tensor]) -> List[Tensor]:
        self.build_task_func()
        assert len(inputs) + len(self.task.outputs) == len(self.task.parameters)
        output_types = [output.data_type for output in self.task.parameters[-len(self.task.outputs):]]
        outputs = [empty(shape=type.const_shape(), dtype=type.scalar_type.name, device=self.device, layout=type.layout) for type in output_types]
        self.pure_run(inputs, outputs)
        return outputs

    def lazy_run(self) -> List[Tensor]:
        output_nodes = self.task.parameters[-len(self.task.outputs):]
        output_types = [output_node.data_type for output_node in output_nodes]
        outputs = []
        for i, output_type in enumerate(output_types):
            outputs.append(Tensor(
                shape=output_type.const_shape(),
                dtype=output_type.scalar_type.name,
                device=self.device,
                storage=None,
                layout=output_type.layout,
                trace=(self, i)
            ))
        return outputs

    def pure_run(self, inputs: List[Tensor], outputs: List[Tensor]):
        self.build_task_func()
        self.task_func(*inputs, *outputs)

        status = get_last_error()
        if status is not None:
            msg = 'Kernel failed. Error:\n{}'.format(self.name, status)
            raise BackendException(msg)

    def reforward(self, inputs: List[Tensor], update_attributes: Optional[Dict[str, Any]] = None) -> List[Tensor]:
        cls = self.__class__
        if not isinstance(self, Operator) or cls is Operator:
            raise ValueError('Can only reforward operator whose class is a proper class of Operator. Please use .clone')
        attributes = self.attrs.copy()
        if update_attributes is not None:
            attributes.update(update_attributes)
        return cls(*inputs, **attributes).run()

    def clone(self, inputs: List[Tensor], update_attributes: Optional[Dict[str, Any]] = None) -> List[Tensor]:
        cls = self.__class__
        attributes = self.attrs.copy()
        if update_attributes is not None:
            attributes.update(update_attributes)

        new_op = cls.__new__(cls)
        new_op.name = self.name
        new_op.inputs = inputs
        new_op.task = self.task
        new_op.attrs = attributes
        new_op.device = self.device
        new_op.outputs = new_op.run()
        new_op.task_func = None
        return new_op.outputs

    def dummy_inputs(self) -> List[Tensor]:
        dummy_inputs = []
        for x in self.inputs:
            if x.storage is not None:
                dummy_inputs.append(x)
            else:
                if x.dtype in ['float32', 'float16', 'bfloat16']:
                    dummy_inputs.append(empty_like(x))
                else:
                    raise ValueError('Can not generate dummy input for dtype {}'.format(x.dtype))
        return dummy_inputs

    def dummy_outputs(self) -> List[Tensor]:
        output_types = [output.data_type for output in self.task.parameters[-len(self.task.outputs):]]
        dummy_outputs = [empty(shape=type.const_shape(), dtype=type.scalar_type.name, device='cuda', layout=type.layout) for type in output_types]
        return dummy_outputs

    def latency(self, warmup=3, number=20, repeat=5, median=True) -> Union[List[float], float]:
        from hidet.ffi import cuda
        from time import time
        import numpy as np
        dummy_inputs = self.dummy_inputs()
        outputs = self.dummy_outputs()

        self.imperative_run(dummy_inputs)
        for t in range(warmup):
            self.task_func(*dummy_inputs, *outputs)
        cuda.device_synchronize()
        results = []
        for i in range(repeat):
            cuda.device_synchronize()
            t1 = time()
            for j in range(number):
                self.task_func(*dummy_inputs, *outputs)
            cuda.device_synchronize()
            t2 = time()
            results.append((t2 - t1) / number * 1000.0)
        if median:
            return float(np.median(results))
        return results

    def build_task_func(self):
        from hidet.driver import build_task
        if self.task_func is None:
            pc = _profile_config
            self.task_func = build_task(self.task, space_level=self._current_space_level, target_device=self.device, warmup=pc.warmup, number=pc.number, repeat=pc.repeat, use_cache=self._use_cache)


def space_level(level=0):
    """Set the schedule space level of tunable operator.

    Some operators can be tuned in hidet to achieve the best performance, such as matrix multiplication.

    During tuning, different operator schedules will be tried and profiled to get the best one.

    We call the space of the tried operator schedule `schedule space`. There is a trade-off between the
    tuning time and the operator execution time. If we try more schedules, the tuning process would take
    longer time, and we are likely to find better schedule.

    This function allows user to set the space level that controls the search space we tried.

    By convention, we have space level

    - 0 for schedule space contains only a single schedule.
    - 1 for schedule space contains tens of schedules so that the tuning time will be less than 1 minute.
    - 2 for arbitrary large space.

    Usage

    .. code-block:: python

        hidet.space_level(2)

    After calling above function, all subsequent compilation would use space level 2, until we call this
    function again with another space level.

    Parameters
    ----------
    level: int
        The space level to use. Candidates: 0, 1, and 2.
    """
    Operator._current_space_level = level


def get_space_level() -> int:
    """Get the current space level.

    Returns
    -------
    ret: int
        The current space level.
    """
    return Operator._current_space_level


def cache_operator(use_cache: bool = True):
    """Whether to cache compiled operator.

    By default, hidet would cache all compiled operator and reuse whenever possible.

    If user wants to disable the cache, run

    .. code-block:: python

        hidet.cache_operator(False)

    Parameters
    ----------
    use_cache: bool
        Whether to cache the compiled operator.
    """
    Operator._use_cache = use_cache


def profile_config(warmup=3, number=10, repeat=3):
    """Set the profiling config of operator tuning.

    To profile a schedule, hidet will run the following code:

    .. code-block:: python

        for i in range(warmup):
            run()
        latency = []
        for i in range(repeat):
            synchronize device
            t1 = time()
            for j in range(number):
                run()
            synchronize device
            t2 = time()
            latency.append((t2 - t1) / number)
        return median of latency

    Thus, there will be total ``warmup + number * repeat`` times of execution.

    Parameters
    ----------
    warmup: int
        The number of warmup runs.
    number: int
        The number of runs in a repeat.
    repeat: int
        The number of repeats.
    """
    global _profile_config
    _profile_config = ProfileConfig(warmup, number, repeat)


def get_profile_config() -> ProfileConfig:
    """Get the current profile config.

    Returns
    -------
    ret: ProfileConfig
        The current profile config. It is a named tuple with fields (warmup, number, repeat).
    """
    return _profile_config

