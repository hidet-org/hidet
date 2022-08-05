from typing import List, Optional, Dict, Any, Iterable, Tuple, Union
from collections import defaultdict

from hidet.ir.task import Task
from hidet.runtime import CompiledFunction
from hidet.driver import build_task
from hidet.tos.tensor import empty, empty_like, Tensor


def trim_op_ending(name: str):
    return name[:-2] if name.endswith('Op') else name


class Operator:
    _current_space_level = 0
    _use_cache = True

    _task_cache: Dict[int, Dict[str, CompiledFunction]] = defaultdict(dict)

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
        self.name = name if name else trim_op_ending(self.__class__.__name__)

        assert all(isinstance(v, Tensor) for v in inputs)

        # cache
        self.task_func: Optional[CompiledFunction] = None

    def __str__(self):
        arguments = ['{}: {}{}'.format(i, t.dtype, t.shape) for i, t in enumerate(self.inputs)]
        attributes = ['{}={}'.format(name, str(value)) for name, value in self.attrs.items()]
        return '{}({})'.format(self.name, ', '.join(arguments + attributes))

    def __dir__(self) -> Iterable[str]:
        return ['task', 'inputs', 'outputs', 'attributes', 'name'] + list(self.attrs)

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
        outputs = [empty(shape=type.const_shape(), dtype=type.scalar_type.name, device='cuda', layout=type.layout) for type in output_types]
        self.pure_run(inputs, outputs)
        return outputs

    def lazy_run(self) -> List[Tensor]:
        output_types = [output.data_type for output in self.task.parameters[-len(self.task.outputs):]]
        outputs = [Tensor(shape=type.const_shape(), dtype=type.scalar_type.name, device='cuda', storage=None, layout=type.layout, trace=(self, i)) for i, type in enumerate(output_types)]
        return outputs

    def pure_run(self, inputs: List[Tensor], outputs: List[Tensor]):
        self.build_task_func()
        self.task_func(*inputs, *outputs)

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
        if self.task_func is None:
            task_string = str(self.task)
            level = self._current_space_level
            if task_string in self._task_cache[level]:
                self.task_func = self._task_cache[level][task_string]
            else:
                self.task_func = build_task(self.task, space_level=self._current_space_level, use_cache=self._use_cache)
                self._task_cache[level][task_string] = self.task_func


def space_level(level=0):
    Operator._current_space_level = level


def get_space_level() -> int:
    return Operator._current_space_level


def cache_operator(use_cache=True):
    Operator._use_cache = use_cache
