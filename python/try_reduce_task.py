import hidet
from hidet import Task
from hidet.ir.compute import tensor_input, compute, reduce
from typing import List

def run_task(task: Task, inputs: List[hidet.Tensor], outputs: List[hidet.Tensor]):
    """Run given task and print inputs and outputs"""
    from hidet.runtime import CompiledFunction

    # build the task
    func: CompiledFunction = hidet.driver.build_task(task, target_device='cpu')
    params = inputs + outputs

    # run the compiled task
    func(*params)

    print('Task:', task.name)
    print('Inputs:')
    for tensor in inputs:
        print(tensor)
    print('Output:')
    for tensor in outputs:
        print(tensor)
    print()


def reduce_sum_example():
    a = tensor_input('a', dtype='float32', shape=[4, 3])
    b = compute(
        'b',
        shape=[4],
        fcompute=lambda i: reduce(
            shape=[3], fcompute=lambda j: a[i, j], reduce_type='sum'
        ),
    )
    task = Task('reduce_sum', inputs=[a], outputs=[b])
    run_task(task, [hidet.randn([4, 3])], [hidet.empty([4])])


reduce_sum_example()
