from hidet.implement.common import expand_loop
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.expr import Var
from hidet.ir.func import IRModule, Function
from hidet.ir.task import Host
from hidet.tos.task import Task


def generic_cpu_schedule(task: Task) -> IRModule:
    assert len(task.outputs) == 0
    func_param_vars = [Var(param.name, param.data_type) for param in task.parameters]
    input_map = {p: v for p, v in zip(task.parameters, func_param_vars)}
    body, _, new_buffer_map = expand_loop(task.outputs[0], input_map)
    func_locals = list(new_buffer_map.values())
    func = Function(task.name + '.host', params=func_param_vars, body=body, ret_type=VoidType(),
                    local_vars=func_locals, local_const_vars=[], attrs={'worker': Host()})
    module = IRModule({func.name: func})
    return module
