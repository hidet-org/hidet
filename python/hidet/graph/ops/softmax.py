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
                or (self.axis != len(self.x_shape) - 1 and self.axis != -1)\
                or self.inputs[0].type.dtype != float32:  # not row-major, avx no good not fp32, need diff intrinsics
            return NotImplemented  # use auto-scheduler
        # return NotImplemented
        # return self.schedule_softmax_cpu()
        return tune.extract_ir_modules(self.schedule_softmax_cpu)

    @tune.space(2, nthreads=[4, 8, 16, 32, 64, 96])
    @tune.space(1, nthreads=[8, 16])
    def schedule_softmax_cpu(self, nthreads=16) -> IRModule:
        import hidet
        from hidet.ir.primitives.cpu.avx import avx_f32x8_subtract, avx_f32x8_load, avx_f32x8_setzero, avx_f32x8_store,\
            avx_f32x8_add, avx_f32x8_max, avx_f32x8_permute, avx_f32x8_permute_2f128, avx_f32x8_extract_last, \
            avx_f32x8_extract_half, avx_f32x4_add, avx_f32x4_hadd, avx_f32x4_extract_last, avx_f32x8_broadcast, \
            avx_f32x8_divide, avx_f32x8_to_i32x8, avx_i32x8_to_f32x8, avx_i32x8_broadcast, avx_i32x8_add, \
            avx_i32x8_bitwiseand, avx_f32x8_fmadd, avx_f32x8_multiply, avx_i32x8_greaterthan, avx_i32x8_leftshift_imm
        from hidet.ir.dtypes import float32x8
        from hidet.lang import tensor
        from hidet.ir.stmt import DeclareScope
        from hidet.lang import grid
        from hidet.lang.mapping import spatial
        import numpy as np
        row_size, col_size = 1, self.x_shape[-1]
        head = []
        head_size = 1
        shape = self.inputs[0].shape
        if len(self.x_shape) != 1:
            row_size, col_size = self.x_shape[-2], self.x_shape[-1]
            head = shape[:-2]
            head_size = np.prod(np.array(head))
        matrix_size = row_size * col_size

        with hidet.script_module() as module:

            @hidet.script
            def avx_poly_eval_7(x: float32x8, c0: float32x8, c1: float32x8, c2: float32x8, c3: float32x8, c4: float32x8,
                                c5: float32x8, c6: float32x8, c7: float32x8):
                x2 = avx_f32x8_multiply(x, x)
                x4 = avx_f32x8_multiply(x2, x2)
                return avx_f32x8_fmadd(avx_f32x8_fmadd(avx_f32x8_fmadd(c7, x, c6), x2, avx_f32x8_fmadd(c5, x, c4)), x4,
                                       avx_f32x8_fmadd(avx_f32x8_fmadd(c3, x, c2), x2, avx_f32x8_fmadd(c1, x, c0)))

            @hidet.script
            def avx_exp(x: float32x8) -> float32x8:
                MASK = avx_i32x8_broadcast(0x7FFFFFFF)
                ARG_MAX = avx_i32x8_broadcast(0x42AE0000)
                tbl_ln2 = float.fromhex('0x1.71547652b82fep+0')
                TBL_LN2 = avx_f32x8_broadcast(~tbl_ln2)
                exp_huge = float.fromhex('0x1.8p+23')
                EXP_HUGE = avx_f32x8_broadcast(~exp_huge)
                ln2_tbl_h = float.fromhex('0x1.63p-1')
                LN2_TBL_H = avx_f32x8_broadcast(~ln2_tbl_h)
                ln2_tbl_t = float.fromhex('-0x1.bd0104p-13')
                LN2_TBL_T = avx_f32x8_broadcast(~ln2_tbl_t)
                EXPF_BIAS = avx_i32x8_broadcast(127)

                c0 = float.fromhex("0x1p0")
                C0 = avx_f32x8_broadcast(~c0)
                c1 = float.fromhex("0x1p-1")
                C1 = avx_f32x8_broadcast(~c1)
                c2 = float.fromhex("0x1.555554p-3")
                C2 = avx_f32x8_broadcast(~c2)
                c3 = float.fromhex("0x1.555468p-5")
                C3 = avx_f32x8_broadcast(~c3)
                c4 = float.fromhex("0x1.1112fap-7")
                C4 = avx_f32x8_broadcast(~c4)
                c5 = float.fromhex("0x1.6da4acp-10")
                C5 = avx_f32x8_broadcast(~c5)
                c6 = float.fromhex("0x1.9eb724p-13")
                C6 = avx_f32x8_broadcast(~c6)

                vx = avx_f32x8_to_i32x8(x)
                vx = avx_i32x8_bitwiseand(vx, MASK)
                cond = avx_i32x8_greaterthan(vx, ARG_MAX)
                # if cond != 0:
                    # scalar exp
                z = avx_f32x8_multiply(x, TBL_LN2)
                dn = avx_f32x8_add(z, EXP_HUGE)
                n = avx_f32x8_to_i32x8(dn)
                r1 = avx_f32x8_subtract(x, (avx_f32x8_multiply(dn, LN2_TBL_H)))
                r2 = avx_f32x8_multiply(dn, LN2_TBL_T)
                r = avx_f32x8_subtract(r1, r2)
                m = avx_i32x8_leftshift_imm(avx_i32x8_add(n, EXPF_BIAS), 23)  # implement bitshift
                r2 = avx_f32x8_multiply(r, r)
                r4 = avx_f32x8_multiply(r2, r2)
                poly = avx_f32x8_fmadd(avx_f32x8_fmadd(avx_f32x8_fmadd(C6, r, C5), r2, avx_f32x8_fmadd(C4, r, C3)), r4,
                                       avx_f32x8_fmadd(avx_f32x8_fmadd(C2, r, C1), r2, avx_f32x8_fmadd(C0, r, C0)))
                result = avx_f32x8_multiply(poly, avx_i32x8_to_f32x8(m))

                return result

            @hidet.script
            def find_max(max_vec: float32x8) -> float32:
                y = avx_f32x8_permute_2f128(max_vec, max_vec, 1)  # swap first and last 4
                m1 = avx_f32x8_max(max_vec, y)
                m2 = avx_f32x8_permute(m1, 0b01001110)  # reshuffle to 2 elems per vec and compare
                m3 = avx_f32x8_max(m1, m2)
                m4 = avx_f32x8_permute(m3, 0b10110001)  # reshuffle to 1 elem per vec and compare
                m = avx_f32x8_max(m3, m4)  # max val
                return avx_f32x8_extract_last(m)

            @hidet.script
            def find_sum(x: float32x8) -> float32:
                sum_vec = avx_f32x4_add(avx_f32x8_extract_half(x, 0b0), avx_f32x8_extract_half(x, 0b1))
                sum_vec = avx_f32x4_hadd(sum_vec, sum_vec)
                sum_vec = avx_f32x4_hadd(sum_vec, sum_vec)
                return avx_f32x4_extract_last(sum_vec)

            @hidet.script
            def softmax_cpu_kernel(x: float32[shape], out: float32[shape]):
                # can pass shape = x.shape, float32[shape]
                for k in range(head_size):
                    offset = matrix_size * k
                    head_idx = spatial(*head).map(k)
                    para = 'p' + str(nthreads)
                    for i in grid(row_size, attrs=para):
                        # find max
                        max_val = x[head_idx][i][0]
                        if col_size >= 8:
                            max_vec = avx_f32x8_load(x + offset + i * col_size)
                            for j in range(col_size // 8):
                                data_vec = avx_f32x8_load(x + offset + i * col_size + j * 8)
                                max_vec = avx_f32x8_max(max_vec, data_vec)
                            max_val = find_max(max_vec)
                        for j in range(col_size % 8):
                            max_val = max_val if max_val > x[head_idx][i][col_size - col_size % 8 + j] \
                                else x[head_idx][i][col_size - col_size % 8 + j]

                        # subtract max, take exp and find exp sum
                        sum_value = 0.0
                        if col_size >= 8:
                            sum_exp_vec = avx_f32x8_setzero()
                            max_vec = avx_f32x8_broadcast(~max_val)
                            for j in range(col_size // 8):
                                val_vec = avx_f32x8_load(x + offset + i * col_size + j * 8)
                                val_vec = avx_f32x8_subtract(val_vec, max_vec)
                                # apply exponent val_vec = avxexponent
                                arr = tensor(scope=DeclareScope.Default, dtype=float32, shape=[8])
                                avx_f32x8_store(arr, val_vec)
                                for n in range(8):
                                    arr[n] = prim.exp(arr[n])
                                val_vec = avx_f32x8_load(arr)
                                # val_vec = avx_exp(val_vec)
                                avx_f32x8_store(out + offset + i * col_size + j * 8, val_vec)
                                sum_exp_vec = avx_f32x8_add(sum_exp_vec, val_vec)
                            sum_value = find_sum(sum_exp_vec)
                        for j in range(col_size % 8):
                            out[head_idx][i][col_size - col_size % 8 + j] = \
                                prim.exp(x[head_idx][i][col_size - col_size % 8 + j] - max_val)
                            sum_value += out[head_idx][i][col_size - col_size % 8 + j]

                        # divide by exp sum
                        if col_size >= 8:
                            # divide
                            sum_vec8 = avx_f32x8_broadcast(~sum_value)
                            # avx_exp(sum_vec8)
                            for j in range(col_size // 8):
                                avx_f32x8_store(out + offset + i * col_size + j * 8,
                                                avx_f32x8_divide(avx_f32x8_load(out + offset + i * col_size + j * 8),
                                                                 sum_vec8))
                        for j in range(col_size % 8):
                            out[head_idx][i][col_size - col_size % 8 + j] = \
                                out[head_idx][i][col_size - col_size % 8 + j] / sum_value

            softmax_cpu_kernel.kind = "cpu_kernel"
            find_max.kind = "cpu_internal"
            find_sum.kind = "cpu_internal"
            # avx_exp.kind = "cpu_internal"
            # avx_poly_eval_7.kind = "cpu_internal"
            assert isinstance(softmax_cpu_kernel, hidet.ir.Function)
            ir_module = module.ir_module()
            return ir_module

# sum = _mm_add_ps(_mm256_extractf128_ps(vector, 0), _mm256_extractf128_ps(vector, 1));
# sum = _mm_hadd_ps(sum, sum);
# sum = _mm_hadd_ps(sum, sum);
# return _mm_cvtss_f32(sum);

# __m256 y = _mm256_permute2f128_ps(x, x, 1); // 8 5 3 6 8 5 3 6
# __m256 m1 = _mm256_max_ps(x, y); // 8 7 3 6 8 5 3 6
# __m256 m2 = _mm256_permute_ps(m1, 0b01001110); // swap 2, 3 and 0, 1, 3 6 8 7 8 5 3 6
# __m256 m3 = _mm256_max_ps(m1, m2); // 8 7 8 7 8 5 3 6
# __m256 m4 = _mm256_permute_ps(m3, 0b10110001); // 7 8 8 7 8 5 3 6
# __m256 m = _mm256_max_ps(m3, m4); // max elem will be available in all elements of m



            # @hidet.script
            # def avx_poly_eval_7(x: float32x8, c0: float32x8, c1: float32x8, c2: float32x8, c3: float32x8, c4: float32x8,
            #                     c5: float32x8, c6: float32x8, c7: float32x8):
            #     x2 = avx_f32x8_multiply(x, x)
            #     x4 = avx_f32x8_multiply(x2, x2)
            #     return avx_f32x8_fmadd(avx_f32x8_fmadd(avx_f32x8_fmadd(c7, x, c6), x2, avx_f32x8_fmadd(c5, x, c4)), x4,
            #                            avx_f32x8_fmadd(avx_f32x8_fmadd(c3, x, c2), x2, avx_f32x8_fmadd(c1, x, c0)))
            #
            # @hidet.script
            # def avx_exp(x: float32x8) -> float32x8:
            #     MASK = avx_i32x8_broadcast(0x7FFFFFFF)
            #     ARG_MAX = avx_i32x8_broadcast(0x42AE0000)
            #     tbl_ln2 = float.fromhex('0x1.71547652b82fep+0')
            #     TBL_LN2 = avx_f32x8_broadcast(~tbl_ln2)
            #     exp_huge = float.fromhex('0x1.8p+23')
            #     EXP_HUGE = avx_f32x8_broadcast(~exp_huge)
            #     ln2_tbl_h = float.fromhex('0x1.63p-1')
            #     LN2_TBL_H = avx_f32x8_broadcast(~ln2_tbl_h)
            #     ln2_tbl_t = float.fromhex('-0x1.bd0104p-13')
            #     LN2_TBL_T = avx_f32x8_broadcast(~ln2_tbl_t)
            #     EXPF_BIAS = avx_i32x8_broadcast(127)
            #
            #     c0 = float.fromhex("0x1p0")
            #     C0 = avx_f32x8_broadcast(~c0)
            #     c1 = float.fromhex("0x1p-1")
            #     C1 = avx_f32x8_broadcast(~c1)
            #     c2 = float.fromhex("0x1.555554p-3")
            #     C2 = avx_f32x8_broadcast(~c2)
            #     c3 = float.fromhex("0x1.555468p-5")
            #     C3 = avx_f32x8_broadcast(~c3)
            #     c4 = float.fromhex("0x1.1112fap-7")
            #     C4 = avx_f32x8_broadcast(~c4)
            #     c5 = float.fromhex("0x1.6da4acp-10")
            #     C5 = avx_f32x8_broadcast(~c5)
            #     c6 = float.fromhex("0x1.9eb724p-13")
            #     C6 = avx_f32x8_broadcast(~c6)
            #
            #     vx = avx_f32x8_to_i32x8(x)
            #     vx = avx_i32x8_bitwiseand(vx, MASK)
            #     cond = avx_i32x8_greaterthan(vx, ARG_MAX)
            #     if cond != 0:
            #         # scalar exp
            #     z = avx_f32x8_multiply(x, TBL_LN2)
            #     dn = avx_f32x8_add(z, EXP_HUGE)
            #     n = avx_f32x8_to_i32x8(dn)
            #     r1 = avx_f32x8_subtract(x, (avx_f32x8_multiply(dn, LN2_TBL_H)))
            #     r2 = avx_f32x8_multiply(dn, LN2_TBL_T)
            #     r = avx_f32x8_subtract(r1, r2)
            #     m = avx_i32x8_leftshift_imm(avx_i32x8_add(n, EXPF_BIAS), 23)  # implement bitshift
            #     r2 = avx_f32x8_multiply(r, r)
            #     r4 = avx_f32x8_multiply(r2, r2)
            #     poly = avx_f32x8_fmadd(avx_f32x8_fmadd(avx_f32x8_fmadd(C6, r, C5), r2, avx_f32x8_fmadd(C4, r, C3)), r4,
            #                            avx_f32x8_fmadd(avx_f32x8_fmadd(C2, r, C1), r2, avx_f32x8_fmadd(C0, r, C0)))
            #     result = avx_f32x8_multiply(poly, avx_i32x8_to_f32x8(m))
            #
            #     return result
