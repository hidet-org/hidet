from typing import List
from hidet.ir.func import IRModule
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives import active_mask, shfl_down_sync, shfl_sync
from hidet.ir.stmt import AssignStmt, Stmt
from hidet.tos.task import Task
from hidet.utils import TableBuilder


def warp_reduce(v, op) -> Stmt:
    sb = StmtBuilder()
    with sb.let('mask', active_mask()) as mask:
        for delta in [16, 8, 4, 2, 1]:
            sb += AssignStmt(v, op(v, shfl_down_sync(mask, v, delta=delta)))
        sb += AssignStmt(v, shfl_sync(mask, v, src_lane=0))
    return sb.finish()


def dummy_inputs_from_task(task: Task):
    from hidet.ir.type import TensorType
    from hidet.ir.expr import Constant
    from hidet.tos.tensor import randn
    inputs = []
    for idx, param in enumerate(task.parameters):
        param_type = param.data_type
        assert isinstance(param_type, TensorType)
        assert all(isinstance(s, Constant)for s in param_type.shape)
        stype = param_type.scalar_type.name
        scope = param_type.scope.name
        shape = [int(s) for s in param_type.shape]
        # strides = [int(s) for s in param_type.strides]
        scope2device = {
            'global': 'cuda',
            'host': 'cpu'
        }
        inputs.append(randn(shape, stype, device=scope2device[scope]))
    return inputs


def resolve_ir_modules(ir_modules: List[IRModule], schedules, task: Task, task_label, parallel: bool = True, verbose: bool = True):
    # from hidet.runtime.value import dummy_inputs_from_task
    from hidet.backend import BuildInstance, batch_build
    import numpy as np
    assert len(schedules) == len(ir_modules)
    if len(ir_modules) == 1:
        return ir_modules[0]
    build_instances = [BuildInstance(ir_module=ir_module,
                                     output_dir=f'./outs/resolve/{task_label}/{idx}',
                                     keep_ir=False,
                                     nvcc_keep=False,
                                     verbose=False) for idx, ir_module in enumerate(ir_modules)]
    compiled_modules = batch_build(build_instances, parallel=parallel, verbose=verbose)
    dummy_inputs = dummy_inputs_from_task(task)
    best_latency = None
    best_ir_module = None
    latencies = []
    for schedule, ir_module, compiled_module in zip(schedules, ir_modules, compiled_modules):
        repeat_latency = compiled_module[task.name].profile(*dummy_inputs, warmup=2, number=1, repeat=10)
        latency = float(np.median(repeat_latency))
        latencies.append(latency)
        if best_latency is None or best_latency > latency:
            best_latency = latency
            best_ir_module = ir_module
    with TableBuilder(headers=['idx'] + [v[0] for v in (schedules[0].keys() + schedules[0].derived_keys())] + ['latency']) as tb:
        rows = []
        for idx, (schedule, latency) in enumerate(zip(schedules, latencies)):
            row = [idx] + [v[1] for v in schedule.keys() + schedule.derived_keys()] + [latency]
            rows.append(row)
        rows = sorted(rows, key=lambda v: v[-1])
        for row in rows:
            tb += row
    with open(f'./outs/resolve/{task_label}/report.txt', 'w') as f:
        f.write(str(tb))
    return best_ir_module
