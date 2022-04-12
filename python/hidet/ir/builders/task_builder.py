from typing import Sequence, Optional, Union

from hidet.ir.expr import Expr, Var, Call
from hidet.ir.func import IRModule
from hidet.ir.task import Task, Worker
from hidet.ir.type import TypeNode
from hidet.ir.dialects.compute import TensorInput, ScalarInput, TensorCompute, ReduceCompute, ComputeNode


class TaskBuilder:
    def __init__(self, name: str, worker: Worker, parent_module: IRModule, try_first: Optional[Union[str, Sequence[str]]] = None):
        self.name = name
        self.computation = None
        self.params = []
        self.worker = worker
        self.parent_module = parent_module
        self.func_var = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finish()
        else:
            # there is an exception, propagate it to outer scope
            pass

    def __call__(self, *args):
        assert len(args) == len(self.params)
        if self.func_var is None:
            self.finish()
        return Call(self.func_var, args)

    def extend_params(self, params: Sequence[ComputeNode], types: Sequence[TypeNode]=None):
        self.params.extend(params)
        assert types is None

    def append_param(self, param: ComputeNode, param_type: TypeNode=None):
        self.params.append(param)
        assert param_type is None

    def set_computation(self, computation: Expr):
        assert self.computation is None
        self.computation = computation

    def finish(self):
        from hidet.implement.implementer import implement, impl_context
        assert self.func_var is None
        assert self.computation is not None
        task = Task(self.name, self.computation, self.params, self.worker)
        with impl_context():
            sub_module = implement(task)
        self.parent_module.include(sub_module)
        self.func_var = self.parent_module.lookup_var(self.name)
