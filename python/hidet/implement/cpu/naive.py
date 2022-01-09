from typing import Mapping

from hidet.ir.node import Node
from hidet.ir.expr import Var
from hidet.ir.task import Task, Host
from hidet.ir.func import IRModule, Function
from hidet.ir.dialects.pattern import TaskPattern
from hidet.ir.dialects.lowlevel import VoidType
from hidet.implement.common import expand_loop
from hidet.implement.implementer import Implementer, register_impl


@register_impl('cpu_naive_implementer')
class CpuNaiveImplementer(Implementer):
    def __init__(self):
        self.pattern = TaskPattern(worker=Host())

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        func_param_vars = [Var(param.name, tp) for param, tp in zip(task.params, task.params_type)]
        input_map = {p: v for p, v in zip(task.params, func_param_vars)}
        body, _, new_buffer_map = expand_loop(task.compute, input_map)
        func_locals = new_buffer_map.values()
        func = Function(task.name + '.host', func_param_vars, body, VoidType(), func_locals, {'worker': Host()})
        module = IRModule({func.name: func})
        return module

