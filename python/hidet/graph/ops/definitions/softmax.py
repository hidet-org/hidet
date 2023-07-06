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
from hidet.ir.func import IRModule
from hidet.ir import primitives as prim
from hidet.ir.expr import is_constant
from .utils import Task, TensorNode, compute, reduce
from typing import List, Union
from hidet.ir.dtypes import float32
from hidet.graph.ops.definitions.utils import tune


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
        from hidet.graph.ops.schedules import softmax_cuda_schedule

        if not all(is_constant(dim) for dim in self.inputs[0].shape):
            return NotImplemented  # use auto-scheduler

        return softmax_cuda_schedule(self)

    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        if not all(is_constant(dim) for dim in self.inputs[0].shape):
            return NotImplemented  # use auto-scheduler
        print(self.__dict__)
        print(type(self.outputs[0]), self.outputs[0])
        print(self.x_shape)
        return self.schedule_softmax_cpu()
        return NotImplemented
        return tune.extract_ir_modules(self.schedule_softmax_cpu)

    # @tune.space(2, 'nthreads', [4, 8, 16, 32, 64, 96])
    # @tune.space(1, 'nthreads', [8, 16])
    def schedule_softmax_cpu(self, nthreads=16) -> IRModule:
        import hidet
        from hidet.ir.primitives.cpu.avx import avx_f32x8_subtract, avx_f32x8_load, avx_f32x8_setzero, avx_f32x8_store,\
            avx_f32x8_add, avx_f32x8_max, avx_f32x8_permute, avx_f32x8_permute_2f128, avx_f32x8_extract_one,\
            avx_f32x8_extract_half, avx_f32x4_add, avx_f32x4_hadd, avx_f32x4_extract_one
        from hidet.ir.dtypes import float32x8
        from hidet.lang.constructs.type import tensor
        from hidet.ir.stmt import DeclareScope
        from hidet.lang import grid
        assert len(self.x_shape) == 2, "only test with 2d arr"
        row_size, col_size = self.x_shape[-2], self.x_shape[-1]
        with hidet.script_module() as module:
            @hidet.script
            def find_max(max_vec: float32x8, out_vec: float32[8]):
                y = avx_f32x8_permute_2f128(max_vec, max_vec, 1)
                m1 = avx_f32x8_max(max_vec, y)
                m2 = avx_f32x8_permute(m1, 0b01001110)
                m3 = avx_f32x8_max(m1, m2)
                m4 = avx_f32x8_permute(m3, 0b10110001)
                m = avx_f32x8_max(m3, m4)
                avx_f32x8_store(out_vec, m)
                # return avx_f32x8_extract_lower(m)

            @hidet.script
            def softmax_cpu(x: float32[row_size, col_size], out: float32[row_size, col_size]):
                # TODO: look at the code in softmaxavx.cpp and implement here
                # TODO: look at bookmarked stack overflow avx exp and ask yaoyao about writing C for hidet
                para = 'p' + str(nthreads)
                # x_ptr = x
                for i in grid(row_size, attrs=para):
                    max_val = x[i, 0]
                    if col_size >= 8:
                        max_vec = avx_f32x8_load(x + i * col_size)  # only if greater than equal 8
                        for j in range(col_size//8):
                            data_vec = avx_f32x8_load(x + i * col_size + j * 8)
                            max_vec = avx_f32x8_max(max_vec, data_vec)
                        y = avx_f32x8_permute_2f128(max_vec, max_vec, 1)
                        m1 = avx_f32x8_max(max_vec, y)
                        m2 = avx_f32x8_permute(m1, 0b01001110)
                        m3 = avx_f32x8_max(m1, m2)
                        m4 = avx_f32x8_permute(m3, 0b10110001)
                        m = avx_f32x8_max(m3, m4)
                        max_val = avx_f32x8_extract_one(m)
                    for j in range(col_size % 8):
                        max_val = max_val if max_val > x[i, j] else x[i, j]

                    sum_value = 0.0
                    if col_size >= 8:
                        sum_exp_vec = avx_f32x8_setzero()
                        for j in range(col_size//8):
                            val_vec = avx_f32x8_load(x + i * col_size + j * 8)
                            val_vec = avx_f32x8_subtract(val_vec, m)
                            #apply exponent val_vec = avxexponent
                            avx_f32x8_store(val_vec, out + i * col_size + j * 8)
                            sum_exp_vec = avx_f32x8_add(sum_exp_vec, val_vec)
                        sum_vec = avx_f32x4_add(avx_f32x8_extract_half(sum_exp_vec, 0b0),
                                                avx_f32x8_extract_half(sum_exp_vec, 0b1))
                        sum_vec = avx_f32x4_hadd(sum_vec, sum_vec)
                        sum_vec = avx_f32x4_hadd(sum_vec, sum_vec)
                        sum_value = avx_f32x4_extract_one(sum_vec)
                    for j in range(col_size % 8):
                        out[i, j] = prim.exp(x[i, j])
                        sum_value += out[i, j]

                    # vec = tensor(scope=DeclareScope.Default, dtype=float32, shape=[])
                    # find_max(max_vec, vec)
                    # max_val = vec[0]
                    # out[i, 0] = max_val

                # for i in range(row_size):
                #     max_val = x[i, 0]
                #     for j in range(col_size):
                #         max_val = x[i, j] if max_val < x[i, j] else max_val
                #     out[i, 0] = max_val
            softmax_cpu.kind = "cpu_kernel"
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
