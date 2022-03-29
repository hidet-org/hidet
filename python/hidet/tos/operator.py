from typing import List, Optional, Dict
from hidet.ir.task import Task
from hidet.runtime import CompiledFunction
from hidet.driver import build_task
from hidet.tos.tensor import empty


class Operator:
    imperative_mode = 1
    imperative_opt_mode = 2
    lazy_mode = 3

    current_mode = imperative_mode

    task_cache: Dict[str, CompiledFunction] = {}

    def __init__(self, inputs: List['Tensor'], task: Task):
        from hidet.tos.tensor import Tensor
        self.inputs: List[Tensor] = inputs
        self.task: Task = task
        self.outputs: List[Tensor] = []
        if self.current_mode == self.imperative_mode:
            task_string = str(task)
            # self.outputs.extend(imperative_run(task, inputs, 0, 0))
            if task_string in self.task_cache:
                func = self.task_cache[task_string]
            else:
                func = build_task(task, space_level=0, opt_level=0, use_cache=True)
                self.task_cache[task_string] = func
            output_type = task.type_of_param(task.compute)
            self.outputs.append(empty(shape=[int(v) for v in output_type.shape], dtype=output_type.scalar_type.name, layout=output_type.layout))
            func(*self.inputs, *self.outputs)
        elif self.current_mode == self.lazy_mode:
            output_type = task.type_of_param(task.compute)
            self.outputs.append(empty(shape=[int(v) for v in output_type.shape], dtype=output_type.scalar_type.name, layout=output_type.layout))
        else:
            raise NotImplementedError('coming soon')

    @classmethod
    def set_execution_mode(cls, mode):
        assert mode in [cls.imperative_mode,
                        cls.imperative_opt_mode,
                        cls.lazy_mode]
        cls.current_mode = mode


def imperative_mode(opt=False):
    Operator.set_execution_mode(Operator.imperative_mode)


def lazy_mode():
    Operator.set_execution_mode(Operator.lazy_mode)

