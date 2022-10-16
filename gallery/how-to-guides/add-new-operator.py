from hidet.ir import Task, TensorNode
from hidet.ir.dialects.compute import compute, reduce, tensor_input
from hidet.graph import Operator

# schedule with hidet auto-scheduler


class BatchMatmulTask(Task):
    def __init__(self, a: TensorNode, b: TensorNode):
        batch_size, m_size, k_size = a.const_shape()
        batch_size, k_size, n_size = b.const_shape()

        c = compute(
            name='c',
            shape=[batch_size, m_size, n_size],
            fcompute=lambda r, i, j: reduce(
                shape=[k_size],
                fcompute=lambda k: a[r, i, k] * b[r, k, j],
                reduce_type='sum'
            ),
            scope='global'
        )
        super().__init__(
            name='matmul',
            inputs=[a, b],
            outputs=[c]
        )


def demo_task():
    batch_size, m_size, n_size, k_size = 8, 1024, 1024, 1024
    a = tensor_input(name='a', base_type='float32', shape=[batch_size, m_size, k_size], scope='global')
    b = tensor_input(name='b', base_type='float32', shape=[batch_size, k_size, n_size], scope='global')
    task = BatchMatmulTask(a, b)
    print(task)


demo_task()

# schedule with dedicated schedule-template

# schedule with more than one kernel

if __name__ == '__main__':
    pass
