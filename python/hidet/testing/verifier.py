import numpy as np
from hidet.ir.task import Task, Grid, Host
from hidet.backend import build
from hidet.implement import implement
from hidet.runtime.value import ScalarValue, TensorValue


def verify(grid_task: Task, dummy_inputs):
    assert isinstance(grid_task.worker, Grid)
    grid_name = grid_task.name
    host_name = grid_name + '_host'
    host_task = Task(host_name, grid_task.compute, grid_task.params, grid_task.params_type, Host())

    grid_module = build(implement(grid_task), './outs/grid')
    host_module = build(implement(host_task), './outs/host')

    grid_inputs = []
    host_inputs = []
    for v in dummy_inputs:
        if isinstance(v, ScalarValue):
            grid_inputs.append(v)
            host_inputs.append(v)
        elif isinstance(v, TensorValue):
            grid_inputs.append(v.to_cuda())
            host_inputs.append(v.to_cpu())
        else:
            raise ValueError()

    grid_module[grid_name](*grid_inputs)
    host_module[host_name](*host_inputs)

    for a, b in zip(host_inputs, grid_inputs):
        if isinstance(a, ScalarValue):
            continue
        a, b = a.to_numpy(), b.to_numpy()
        assert(np.allclose(a, b))

