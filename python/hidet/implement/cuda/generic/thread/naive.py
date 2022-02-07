from typing import Mapping

from hidet.ir.node import Node
from hidet.ir.expr import Var
from hidet.ir.stmt import SeqStmt, AssignStmt
from hidet.ir.task import Task, Thread
from hidet.ir.func import IRModule, Function
from hidet.ir.dialects.pattern import TaskPattern
from hidet.ir.dialects.compute import TensorCompute
from hidet.ir.dialects.lowlevel import VoidType
from hidet.implement.implementer import Implementer, register_impl
from hidet.implement.common import expand_loop


@register_impl('cuda_thread_naive_implementer')
class CudaThreadNaiveImplementer(Implementer):
    """
    Naive implementation of any thread task
    """
    def __init__(self):
        self.pattern = TaskPattern(worker=Thread())

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        assert isinstance(task.worker, Thread)
        func_param_vars = [Var(param.name, tp) for param, tp in zip(task.params, task.params_type)]
        input_map = {p: v for p, v in zip(task.params, func_param_vars)}
        body, val, new_buffer_map = expand_loop(task.compute, input_map)
        # if not isinstance(task.compute, TensorCompute):
        #     body = SeqStmt([body, AssignStmt(input_map[task.compute], val)])
        func_locals = list(new_buffer_map.values())
        func = Function(task.name, func_param_vars, body, VoidType(), func_locals, {'worker': Thread()})
        module = IRModule({func.name: func})
        return module

