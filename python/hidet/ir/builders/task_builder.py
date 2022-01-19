from typing import Sequence

from hidet.ir.expr import Expr, Var, Call
from hidet.ir.func import IRModule
from hidet.ir.task import Task, Worker
from hidet.ir.type import BaseType


class TaskBuilder:
    def __init__(self, name: str, worker: Worker, parent_module: IRModule):
        self.name = name
        self.computation = None
        self.params = []
        self.params_type = []
        self.worker = worker
        self.parent_module = parent_module
        self.func_var = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def __call__(self, *args):
        assert len(args) == len(self.params)
        if self.func_var is None:
            self.finish()
        return Call(self.func_var, args)

    def extend_params(self, params: Sequence[Var], types: Sequence[BaseType]):
        assert len(params) == len(types)
        self.params.extend(params)
        self.params_type.extend(types)

    def append_param(self, param: Var, param_type: BaseType):
        self.params.append(param)
        self.params_type.append(param_type)

    def set_computation(self, computation: Expr):
        assert self.computation is None
        self.computation = computation

    def finish(self) -> Call:
        from hidet.implement.implementer import implement
        assert self.func_var is None
        assert len(self.params) == len(self.params_type)
        assert self.computation is not None
        task = Task(self.name, self.computation, self.params, self.params_type, self.worker)
        sub_module = implement(task)
        self.parent_module.include(sub_module)
        self.func_var = self.parent_module.lookup_var(self.name)
