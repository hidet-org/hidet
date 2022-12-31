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
from .utils import Task, Operator, Tensor, TensorNode, compute, input_like, normalize_dim, reduce


class SoftmaxTask(Task):
    def __init__(self, x: TensorNode, axis: int):
        self.x_shape = x.const_shape()
        self.axis = axis

        shape = x.const_shape()
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

    def implement_cuda(self, workding_dir: str) -> IRModule:
        from hidet.graph.ops.schedules import softmax_cuda_schedule

        return softmax_cuda_schedule(self)


class SoftmaxOp(Operator):
    def __init__(self, x: Tensor, axis: int = 1):
        axis = normalize_dim(axis, len(x.shape))
        super().__init__(inputs=[x], task=SoftmaxTask(input_like(x, 'x'), axis), attributes={'axis': axis})


def softmax(x: Tensor, axis=1) -> Tensor:
    return SoftmaxOp(x, axis).get_output(0)
