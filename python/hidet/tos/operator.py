from typing import List, Optional, Dict
from hidet.ir.task import Task
from hidet.runtime import CompiledFunction
from hidet.driver import build_task
from hidet.tos.tensor import empty, Tensor


class Operator:
    imperative_mode = 1
    imperative_opt_mode = 2
    lazy_mode = 3

    current_mode = imperative_mode

    task_cache: Dict[str, CompiledFunction] = {}

    def __init__(self, inputs: List[Tensor], task: Task):
        self.inputs: List[Tensor] = inputs
        self.task: Task = task
        self.outputs: List[Tensor] = []
        # run
        self.run()

    def run(self) -> List[Tensor]:
        print('run {}'.format(self.__class__.__name__))
        if self.current_mode == self.imperative_mode:
            self.outputs = self.imperative_run()
        elif self.current_mode == self.lazy_mode:
            self.outputs = self.lazy_run()
        else:
            raise NotImplementedError('coming soon')
        return self.outputs

    def imperative_run(self) -> List[Tensor]:
        task_string = str(self.task)
        if task_string in self.task_cache:
            func = self.task_cache[task_string]
        else:
            func = build_task(self.task, space_level=0, opt_level=0, use_cache=True)
            self.task_cache[task_string] = func
        output_type = self.task.type_of_param(self.task.compute)
        outputs = [empty(shape=[int(v) for v in output_type.shape], dtype=output_type.scalar_type.name, layout=output_type.layout)]
        func(*self.inputs, *outputs)
        return outputs

    def lazy_run(self) -> List[Tensor]:
        output_type = self.task.type_of_param(self.task.compute)
        return [Tensor(shape=[int(v) for v in output_type.shape], dtype=output_type.scalar_type.name, device='cuda', storage=None, layout=output_type.layout, trace=(self, 0))]


def imperative_mode(opt=False):
    Operator.current_mode = Operator.imperative_mode


def lazy_mode():
    Operator.current_mode = Operator.lazy_mode

