from hidet.tos.ops.schedules.common import expand_loop
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.expr import Var
from hidet.ir.func import IRModule, Function
from hidet.ir.task import Task


def generic_cpu_schedule(task: Task, space_level: int = 0) -> IRModule:
    assert len(task.outputs) == 0
    func_param_vars = [Var(param.name, param.data_type) for param in task.parameters]
    input_map = {p: v for p, v in zip(task.parameters, func_param_vars)}
    body, _, new_buffer_map = expand_loop(task.outputs[0], input_map)
    func_locals = list(new_buffer_map.values())
    func = Function(task.name + '.host', kind='host_kernel', params=func_param_vars, body=body, ret_type=VoidType(),
                    local_vars=func_locals, local_const_vars=[])
    module = IRModule({func.name: func})
    return module
