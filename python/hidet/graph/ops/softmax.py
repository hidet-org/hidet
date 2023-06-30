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
from hidet.ir.module import IRModule
from hidet.ir import primitives as prim
from hidet.ir.expr import is_constant
from hidet.ir.stmt import Stmt, AssignStmt
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives import active_mask, shfl_down_sync, shfl_sync
from .utils import Task, TensorNode, compute, reduce


def warp_reduce(v, op) -> Stmt:
    """
    Reduce over the threads in a warp.

    Parameters
    ----------
    v: Var
        The value to reduce. It must be a variable.
    op:
        An binary operator to represent the reducing operator, must be communicative and associative.

    Returns
    -------
    ret: Stmt
        A block statement to finish the reduction. After reduction, the value in each thread in the warp
        has the reduced value.
    """
    sb = StmtBuilder()
    with sb.let('mask', active_mask()) as mask:
        for delta in [16, 8, 4, 2, 1]:
            sb += AssignStmt(v, op(v, shfl_down_sync(mask, v, delta=delta)))
        sb += AssignStmt(v, shfl_sync(mask, v, src_lane=0))
    return sb.finish()


class SoftmaxTask(Task):
    def __init__(self, x: TensorNode, axis: int):
        self.x_shape = x.shape
        self.axis = axis

        shape = x.shape
        axis_extent = shape[axis]
        reduced_shape = shape[:axis] + shape[axis + 1 :]

        # max value
        max_value = compute(
            name='max_value',
            shape=reduced_shape,
            fcompute=lambda *indices: reduce(
                shape=[axis_extent], fcompute=lambda k: x[indices[:axis] + (k,) + indices[axis:]], reduce_type='max'
            ),
        )

        # exp
        exp_value = compute(
            name='exp_value',
            shape=shape,
            fcompute=lambda *indices: prim.exp(x[indices] - max_value[indices[:axis] + indices[axis + 1 :]]),
        )

        # sum
        sum_value = compute(
            name='sum_value',
            shape=reduced_shape,
            fcompute=lambda *indices: reduce(
                shape=[axis_extent],
                fcompute=lambda k: exp_value[indices[:axis] + (k,) + indices[axis:]],
                reduce_type='sum',
            ),
        )

        # out
        out = compute(
            name='out',
            shape=shape,
            fcompute=lambda *indices: exp_value[indices] / sum_value[indices[:axis] + indices[axis + 1 :]],
        )
        super().__init__(name='softmax', inputs=[x], outputs=[out])

    def implement_cuda(self, working_dir: str) -> IRModule:
        if not all(is_constant(dim) for dim in self.inputs[0].shape):
            return NotImplemented  # use auto-scheduler

        import math
        import hidet
        from hidet.lang import register_tensor
        from hidet.lang import attrs

        from hidet.ir.mapping import TaskMapping

        from hidet.lang.cuda import blockIdx, threadIdx

        shape = self.inputs[0].shape
        axis = self.axis
        reduce_extent = shape[axis]
        reduced_shape = shape[:axis] + shape[axis + 1 :]
        n_reduce = math.prod(reduced_shape)
        warp_size = 32
        outer_extent = (reduce_extent + warp_size - 1) // warp_size
        grid_layout = TaskMapping.row_major(reduced_shape)
        other_inds = list(grid_layout.worker2task(blockIdx.x)[0])
        xdtype = self.inputs[0].type.dtype

        def sum_expr(a, b):
            return a + b

        with hidet.script_module() as module:

            @hidet.script
            def softmax_kernel(xs: xdtype[shape], ys: xdtype[shape]):
                attrs.cuda.block_dim = warp_size
                attrs.cuda.grid_dim = n_reduce

                temp = register_tensor(xdtype, shape=[outer_extent])

                rv = -xdtype.max_value

                # compute maximum in the dimension to be softmaxed across
                for k in range(outer_extent):
                    idx = threadIdx.x + k * warp_size
                    if idx < reduce_extent:
                        temp[k] = xs[other_inds[:axis] + [idx] + other_inds[axis:]]
                        rv = prim.max(rv, temp[k])
                warp_reduce(rv, prim.max)

                # exp
                for k in range(outer_extent):
                    temp[k] = prim.exp(temp[k] - rv)

                rv = xdtype.zero
                for k in range(outer_extent):
                    idx = threadIdx.x + k * warp_size
                    if idx < reduce_extent:
                        rv += temp[k]
                warp_reduce(rv, sum_expr)

                for k in range(outer_extent):
                    idx = threadIdx.x + k * warp_size
                    if idx < reduce_extent:
                        ys[other_inds[:axis] + [idx] + other_inds[axis:]] = temp[k] / rv

        assert isinstance(softmax_kernel, hidet.ir.Function)
        ir_module = module.ir_module()

        return ir_module
