import hidet
from hidet.ir.dialects.compute import compute, reduce, TensorInput
from hidet.tos.operators.common import input_like
from hidet.ir.ntask import Task, Scheduler
from hidet.tos.operators.nn.conv.matmul_cuda_scheduler import MatmulScheduler


def matmul_task(a: TensorInput, b: TensorInput) -> Task:
    a_shape = a.const_shape()
    b_shape = b.const_shape()
    assert len(a_shape) == len(b_shape) == 2, a_shape[1] == b_shape[0]
    m_size, n_size, k_size = a_shape[0], b_shape[0], a_shape[1]

    c = compute(
        name='c',
        shape=[m_size, n_size],
        fcompute=lambda i, j: reduce(
            shape=[k_size],
            fcompute=lambda k: a[i, k] * b[k, j],
            reduce_type='sum'
        )
    )

    return Task(
        name='matmul',
        inputs=[a, b],
        outputs=[c],
        parameters=[a, b, c]
    )


def main():
    a = hidet.ones([3, 3])
    b = hidet.ones([3, 3])
    c = hidet.empty([3, 3])
    task = matmul_task(input_like(a, 'a'), input_like(b, 'b'))
    scheduler = MatmulScheduler()
    ir_module = scheduler(task)
    print(ir_module)
    module = hidet.backend.build(ir_module, './outs/matmul')
    func = module['matmul']
    func(a, b, c)
    print(a)
    print(b)
    print(c)


if __name__ == '__main__':
    main()


