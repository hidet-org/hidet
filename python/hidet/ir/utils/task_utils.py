# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Sequence, Optional, List
from hidet.ir.type import TensorType, FuncType, VoidType
from hidet.ir.func import IRModule
from hidet.ir.task import Task


def func_type_from_task(task: Task) -> FuncType:
    """
    Get the function type for the given task.

    Each task will be lowered to an ir module with a packed function.
    The packed function will accept the packed format of parameters of the task.
    This function will return the function type of the un-packed parameters expected by the packed function.

    For example, if a task has inputs: f32[16, 16], f32[16, 8] and output f32[3, 4]
    Then this function would return a FuncType with param_types: [f32[16, 16], f32[16, 8], f32[3, 4]] and ret_type: None

    Parameters
    ----------
    task: Task
        The task to get the function type.

    Returns
    -------
    ret: FuncType
        The function type for the given task.
    """
    param_types: List[TensorType] = [tensor.ttype for tensor in task.parameters]
    return FuncType(param_types=param_types, ret_type=VoidType())


def validate_schedule(task: Task, device: str, dummy_inputs: Optional[Sequence] = None, rtol=1e-5, atol=1e-5) -> bool:
    """
    Validate the correctness of schedule in the given task.

    Parameters
    ----------
    task: Task
        The task to validate.

    device: str
        The target device.

    dummy_inputs: Optional[Sequence[hidet.graph.tensor.Tensor]]
        The dummy inputs to use for validation.
        If None is given, we will generate random inputs for validation.

    rtol: float
        The relative tolerance for validation.

    atol: float
        The absolute tolerance for validation.

    Returns
    -------
    ret: bool
        Whether the schedule for given device is valid.
    """
    # pylint: disable=import-outside-toplevel, too-many-locals
    from numpy import allclose
    from hidet.driver import build_ir_module
    from hidet.graph.ops.schedules.cuda.auto_scheduler import CudaAutoScheduler
    from hidet.graph.ops.schedules.cpu.auto_scheduler import CpuAutoScheduler
    from hidet.runtime import CompiledFunction
    from hidet.graph.tensor import Tensor, randn, zeros, empty, empty_like

    if dummy_inputs is None:
        dummy_inputs = []
        for input_tensor in task.task_graph.input_tensors:
            tensor_type: TensorType = input_tensor.ttype
            if tensor_type.dtype.is_float():
                dummy_inputs.append(randn(tensor_type.const_shape(), tensor_type.dtype.name, device=device))
            else:
                dummy_inputs.append(zeros(tensor_type.const_shape(), tensor_type.dtype.name, device=device))
    else:
        dummy_inputs = list(dummy_inputs)

    actual_outputs: List[Tensor] = [
        empty(output.ttype.const_shape(), output.ttype.dtype.name, device) for output in task.task_graph.output_tensors
    ]
    desire_outputs: List[Tensor] = [empty_like(output) for output in actual_outputs]

    if len(dummy_inputs) != len(task.task_graph.input_tensors):
        raise ValueError("The number of dummy inputs does not match the number of task inputs.")
    device2scheduler = {"cuda": CudaAutoScheduler, "cpu": CpuAutoScheduler}

    ir_module_actual: IRModule = task.implement(device, workding_dir='./outs')
    ir_module_desire: IRModule = device2scheduler[device]().schedule_task(task, device)

    func_actual: CompiledFunction = build_ir_module(ir_module_actual, func_name=task.name)
    func_desire: CompiledFunction = build_ir_module(ir_module_desire, func_name=task.name)

    func_actual(*dummy_inputs, *actual_outputs)
    func_desire(*dummy_inputs, *desire_outputs)

    for actual, desire in zip(actual_outputs, desire_outputs):
        if not allclose(actual.numpy(), desire.numpy(), rtol=rtol, atol=atol):
            return False
    return True
