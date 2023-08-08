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
from typing import List, Union
from hidet.ir.dtypes import float32
from hidet.ir.library import tune


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

    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        if not all(is_constant(dim) for dim in self.inputs[0].shape)\
                or self.inputs[0].type.dtype != float32:
            return NotImplemented  # use auto-scheduler
        return tune.extract_ir_modules(self.schedule_softmax_cpu)

    @tune.space(2, nthreads=['', 4, 8, 16, 32, 64, 96])
    @tune.space(1, nthreads=['', 8, 16])
    def schedule_softmax_cpu(self, nthreads='') -> IRModule:
        import hidet
        from hidet.ir.primitives.cpu.avx import avx_f32x8_subtract, avx_f32x8_load, avx_f32x8_setzero, avx_f32x8_store,\
            avx_f32x8_add, avx_f32x8_max, avx_f32x8_set1, avx_f32x8_divide, avx_f32x8_to_i32x8,\
            avx_i32x8_to_f32x8, avx_i32x8_set1, avx_i32x8_add, avx_i32x8_bitwiseand, avx_f32x8_fmadd,\
            avx_f32x8_multiply, avx_i32x8_greaterthan, avx_i32x8_leftshift_imm, avx_f32x8_find_sum, avx_f32x8_find_max
        from hidet.ir.dtypes import float32x8
        from hidet.lang import tensor
        from hidet.ir.stmt import DeclareScope
        from hidet.lang import grid
        from hidet.lang.mapping import spatial
        from hidet.utils import prod
        shape = self.inputs[0].shape
        head = shape[:self.axis]
        tail = shape[self.axis:] if self.axis == len(shape) - 1 else shape[self.axis + 1:]
        head_size = prod(head)
        tail_size = prod(tail)
        axis_size = int(shape[self.axis])

        with hidet.script_module() as module:
            @hidet.script
            def softmax_cpu_kernel(x: float32[shape], out: float32[shape]):
                # can pass shape = x.shape, float32[shape]
                para = 'p' + str(nthreads)
                for k in grid(head_size, attrs=para):
                    head_idx = spatial(*head).map(k)
                    if self.axis == len(shape) - 1:  # last dim
                        offset = tail_size * k
                        max_val = x[head_idx][0]
                        if tail_size >= 8:
                            # vectorized find max value
                            max_vec = avx_f32x8_load(x + offset)
                            for i in range(tail_size // 8):
                                data_vec = avx_f32x8_load(x + offset + i * 8)
                                max_vec = avx_f32x8_max(max_vec, data_vec)
                            max_val = avx_f32x8_find_max(max_vec)
                        for i in range(tail_size % 8):
                            # max value of remaining unvectorized parts
                            max_val = max_val if max_val > x[head_idx][tail_size - tail_size % 8 + i] \
                                else x[head_idx][tail_size - tail_size % 8 + i]

                        # subtract max, take exp and find exp sum
                        sum_value = 0.0
                        if tail_size >= 8:
                            sum_exp_vec = avx_f32x8_setzero()
                            max_vec = avx_f32x8_set1(max_val)
                            for i in range(tail_size // 8):
                                val_vec = avx_f32x8_load(x + offset + i * 8)
                                val_vec = avx_f32x8_subtract(val_vec, max_vec)
                                # apply exponent val_vec = avxexponent
                                arr = tensor(scope=DeclareScope.Default, dtype=float32, shape=[8])
                                avx_f32x8_store(arr, val_vec)
                                for n in range(8):
                                    arr[n] = prim.exp(arr[n])
                                val_vec = avx_f32x8_load(arr)
                                # val_vec = avx_exp(val_vec)  # TODO: look into avx exp
                                avx_f32x8_store(out + offset + i * 8, val_vec)
                                sum_exp_vec = avx_f32x8_add(sum_exp_vec, val_vec)
                            sum_value = avx_f32x8_find_sum(sum_exp_vec)
                        for i in range(tail_size % 8):
                            out[head_idx][tail_size - tail_size % 8 + i] = \
                                prim.exp(x[head_idx][tail_size - tail_size % 8 + i] - max_val)
                            sum_value += out[head_idx][tail_size - tail_size % 8 + i]

                        # divide by exp sum
                        if tail_size >= 8:
                            # divide
                            sum_vec8 = avx_f32x8_set1(sum_value)
                            # avx_exp(sum_vec8)
                            for i in range(tail_size // 8):
                                avx_f32x8_store(out + offset + i * 8,
                                                avx_f32x8_divide(avx_f32x8_load(out + offset + i * 8),
                                                                 sum_vec8))
                        for i in range(tail_size % 8):
                            out[head_idx][tail_size - tail_size % 8 + i] = \
                                out[head_idx][tail_size - tail_size % 8 + i] / sum_value
                    else:  # not last dim
                        offset = k * tail_size * axis_size
                        for g in range(tail_size // 8):
                            tail_offset = g * 8
                            # TODO: problem is that the avx is going consecutive but needs to skip rows
                            max_vec = avx_f32x8_load(x + offset + tail_offset)
                            for i in range(axis_size):  # softmax over this guy
                                data_vec = avx_f32x8_load(x + offset + tail_offset + tail_size * i)
                                max_vec = avx_f32x8_max(max_vec, data_vec)
                            sum_exp_vec = avx_f32x8_setzero()
                            for i in range(axis_size):
                                val_vec = avx_f32x8_load(x + offset + tail_offset + tail_size * i)
                                val_vec = avx_f32x8_subtract(val_vec, max_vec)
                                arr = tensor(scope=DeclareScope.Default, dtype=float32, shape=[8])
                                avx_f32x8_store(arr, val_vec)
                                for n in range(8):
                                    arr[n] = prim.exp(arr[n])
                                val_vec = avx_f32x8_load(arr)
                                avx_f32x8_store(out + offset + tail_offset + tail_size * i, val_vec)
                                sum_exp_vec = avx_f32x8_add(sum_exp_vec, val_vec)
                            for i in range(axis_size):
                                avx_f32x8_store(out + offset + tail_offset + tail_size * i,
                                                avx_f32x8_divide(avx_f32x8_load(out + offset + tail_offset + tail_size * i),
                                                                 sum_exp_vec))
                        max_arr = tensor(scope=DeclareScope.Default, dtype=float32, shape=[tail_size % 8])
                        for j in range(tail_size % 8):
                            max_arr[j] = 0.0
                        for p in range(axis_size):
                            for j in range(tail_size % 8):
                                last_idx = spatial(*tail).map(tail_size - tail_size % 8 + j)
                                max_arr[j] = prim.max(max_arr[j], x[head_idx][p][last_idx])  # TODO: index
                        sum_exp_arr = tensor(scope=DeclareScope.Default, dtype=float32, shape=[tail_size % 8])
                        for j in range(tail_size % 8):
                            sum_exp_arr[j] = 0.0
                        for p in range(axis_size):
                            for j in range(tail_size % 8):
                                last_idx = spatial(*tail).map(tail_size - tail_size % 8 + j)
                                out[head_idx][p][last_idx] = prim.exp(x[head_idx][p][last_idx] - max_arr[j])
                                sum_exp_arr[j] += out[head_idx][p][last_idx]
                        for p in range(axis_size):
                            for j in range(tail_size % 8):
                                last_idx = spatial(*tail).map(tail_size - tail_size % 8 + j)
                                out[head_idx][p][last_idx] = out[head_idx][p][last_idx] / sum_exp_arr[j]

            softmax_cpu_kernel.kind = "cpu_kernel"
            # avx_exp.kind = "cpu_internal"
            # avx_poly_eval_7.kind = "cpu_internal"
            assert isinstance(softmax_cpu_kernel, hidet.ir.Function)
            ir_module = module.ir_module()
            return ir_module
        