from hidet.ir.func import IRModule
from .utils import Task, Operator, Tensor, TensorNode, compute, input_like, normalize_dim, reduce
from hidet.ir import primitives as prim


class SoftmaxTask(Task):
    def __init__(self, x: TensorNode, axis: int):
        self.x_shape = x.const_shape()
        self.axis = axis

        shape = x.const_shape()
        axis_extent = shape[axis]
        reduced_shape = shape[:axis] + shape[axis+1:]

        # max value
        max_value = compute(
            name='max_value',
            shape=reduced_shape,
            fcompute=lambda *indices: reduce(
                shape=[axis_extent],
                fcompute=lambda k: x[indices[:axis] + (k,) + indices[axis:]],
                reduce_type='max'
            )
        )

        # exp
        exp_value = compute(
            name='exp_value',
            shape=shape,
            fcompute=lambda *indices: prim.exp(x[indices] - max_value[indices[:axis] + indices[axis+1:]])
        )

        # sum
        sum_value = compute(
            name='sum_value',
            shape=reduced_shape,
            fcompute=lambda *indices: reduce(
                shape=[axis_extent],
                fcompute=lambda k: exp_value[indices[:axis] + (k,) + indices[axis:]],
                reduce_type='sum'
            )
        )

        # out
        out = compute(
            name='out',
            shape=shape,
            fcompute=lambda *indices: exp_value[indices] / sum_value[indices[:axis] + indices[axis+1:]]
        )
        super().__init__(
            name='softmax',
            inputs=[x],
            outputs=[out]
        )

    def implement_cuda(self) -> IRModule:
        from hidet.tos.ops.schedules import softmax_cuda_schedule
        return softmax_cuda_schedule(self)

    def fast_implement(self, space_level: int) -> bool:
        return True


class SoftmaxOp(Operator):
    def __init__(self,
                 x: Tensor,
                 axis: int = 1):
        axis = normalize_dim(axis, len(x.shape))
        super().__init__(
            inputs=[x],
            task=SoftmaxTask(input_like(x, 'x'), axis),
            attributes={
                'axis': axis
            }
        )


def softmax(x: Tensor, axis=1) -> Tensor:
    return SoftmaxOp(x, axis).get_output(0)
