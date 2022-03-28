from typing import List
from hidet.ir.task import Task
from hidet.implement import implement
from hidet.backend import build
from hidet.runtime.value import TensorValue
from hidet.driver import build_task

kernel_idx = 0


def imperative_run(task: Task, inputs: List['Tensor'], space_level, opt_level) -> List['Tensor']:
    global kernel_idx
    from hidet.tos.module import Tensor
    for tensor in inputs:
        if tensor.attached_value is None:
            raise ValueError('Please attach a tensor value to given tensor {}.'.format(tensor.name))
    kernel_idx += 1
    ttype = task.type_of_param(task.compute)

    inputs_value = [input.attached_value for input in inputs]
    outputs_value = [TensorValue.zeros(shape=[int(v) for v in ttype.shape], scalar_type=ttype.scalar_type.name, scope='global')]
    # ir_module = implement(task)
    # module = build(ir_module, output_dir=f'./outs/kernels/{kernel_idx}', keep_ir=keep_ir, nvcc_keep=nvcc_keep)
    func = build_task(task, space_level=space_level, opt_level=opt_level, use_cache=True)
    func(*inputs_value, *outputs_value)
    return [Tensor(shape=ttype.shape, dtype=ttype.scalar_type.name, layout=ttype.layout, value=outputs_value[0])]
