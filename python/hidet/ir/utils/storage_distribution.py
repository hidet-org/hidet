from typing import List
import operator
import functools
from hidet.ir.expr import Expr, Var, TensorElement, Constant
from hidet.ir.functors import collect, simplify
from hidet.ir.dialects.compute import TensorCompute, ReduceCompute, TensorInput
from hidet.implement.cuda.layout import TaskLayout


def get_storage_distribution(computation: Expr, task_axes: List[Var], task_layout: TaskLayout):
    assert len(task_axes) == len(task_layout.task_shape)

    task_shape = task_layout.task_shape
    task_axes_order = {axis: order for order, axis in enumerate(task_axes)}

    # collect tensor and reduce compute
    tcs: List[TensorCompute] = collect(computation, TensorCompute)
    rcs: List[ReduceCompute] = collect(computation, ReduceCompute)

    # get axis to extent mapping
    axis2extent = {}
    for tc in tcs:
        for i, axis in enumerate(tc.axes):
            axis2extent[axis] = int(simplify(tc.shape[i]))
    for rc in rcs:
        for i, axis in enumerate([rc.axis]):
            axis2extent[axis] = int(simplify(rc.shape[i]))

    # check task axes' extent vs task_layout shape
    for axis, order in task_axes_order.items():
        assert axis in axis2extent
        extent = axis2extent[axis]
        assert extent == task_shape[order]

    # get tensor access
    results = {}
    tes: List[TensorElement] = collect(computation, TensorElement)
    for te in tes:
        ti = te.base
        if not isinstance(ti, TensorInput):
            # we only care about input tensor
            continue
        assert ti not in results  # do not support accessing multiple times, for now
        assert all(isinstance(idx, (Var,)) for idx in te.indices)  # only support simple form of inferring
        te_task_axes = [axis for axis in te.indices if axis in task_axes]

        te_other_axes = [axis for axis in te.indices if axis not in task_axes]
        te_other_axes_extents = [axis2extent[axis] for axis in te_other_axes]
        te_other_axes_base = []
        for i in range(len(te_other_axes)):
            te_other_axes_base.append(functools.reduce(operator.mul, te_other_axes_extents[i+1:], 1))
        length = functools.reduce(operator.mul, te_other_axes_extents)
        # give up this solution, keep fighting!



#
#    A[global_index] -> local_index
#
#    task(i, j) -> A[i, k]
#    A[i, k] <- task(i, j) worker for all j
#    A[i, k] <- [worker_expr(i, j) for j]
#
#    A[i, j, worker] -> A_local[local_index]
#
#    A[0, 0] -> [t0, t2, t4, ..., t14]
#    t0 -> [A[0, 0], A[1, 0], A[2, 0], A[3, 0]]
#
