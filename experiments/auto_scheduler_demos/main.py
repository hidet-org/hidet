import hidet
import hidet.utils.cuda
from hidet.ir.task import Task
from hidet.graph.ops.schedules.cuda.auto_scheduler import CudaAutoScheduler


def old():
    from hidet.graph.tensor import randn, empty
    from hidet.ir.dialects.compute import tensor_input, compute, reduce
    from hidet.driver import build_task
    a = tensor_input('a', 'float32', [100, 100], 'global')
    b = compute('b', [100], lambda i: reduce([100], lambda k: a[i, k], 'sum'))
    c = compute('c', [100, 100], lambda i, j: b[i])
    task = Task(name='ggr', inputs=[a], outputs=[c])
    print(task)
    func = build_task(task, space_level=0, cache_dir='./outs/cache')
    aa, bb = randn([100, 100]), empty([100, 100])
    print(func.profile(aa, bb))


def brand_new():
    from hidet.graph.tensor import randn, empty
    from hidet.ir.dialects.compute import tensor_input, compute, reduce
    from hidet.ir.type import FuncType, TensorType, tensor_type
    from hidet.ir.dialects.lowlevel import VoidType
    from hidet.driver import build_task, build_ir_module
    a = tensor_input('a', 'float32', [100, 100], 'global')
    b = compute('b', [100], lambda i: reduce([100], lambda k: a[i, k], 'sum'))
    c = compute('c', [100, 100], lambda i, j: b[i])
    task = Task(name='ggr', inputs=[a], outputs=[c])
    print(task)

    auto_scheduler = CudaAutoScheduler()
    ir_module = auto_scheduler.schedule_task(task)
    print(ir_module)

    func = build_ir_module(
        ir_module,
        func_name='ggr',
        func_type=FuncType(
            [tensor_type('global', 'float32', [100]), tensor_type('global', 'float32', [100, 100])],
            ret_type=VoidType()
        )
    )


def new_scheduler_example_2():
    from hidet.graph.tensor import randn, empty
    from hidet.ir.dialects.compute import tensor_input, compute, reduce
    from hidet.ir.type import FuncType, TensorType, tensor_type
    from hidet.ir.dialects.lowlevel import VoidType
    from hidet.driver import build_task, build_ir_module
    a = tensor_input('a', 'float32', [100, 100], 'global')
    b = compute('b', [100], lambda i: reduce([100], lambda k: a[i, k], 'sum'))
    task = Task(name='ggr', inputs=[a], outputs=[b])
    print(task)

    auto_scheduler = CudaAutoScheduler()
    ir_module = auto_scheduler.schedule_task(task)
    print(ir_module)

    func = build_ir_module(
        ir_module,
        func_name='ggr',
        func_type=FuncType(
            [tensor_type('global', 'float32', [100, 100]), tensor_type('global', 'float32', [100])],
            ret_type=VoidType()
        )
    )
    aa = hidet.randn([100, 100])
    bb = hidet.empty([100])
    print(aa)
    print(bb)
    func(aa, bb)
    print(bb)
    hidet.utils.cuda.device_synchronize()


def demo_cpu_auto_scheduler():
    import hidet
    a = hidet.randn([3, 4], device='cpu')
    # b = hidet.ops.layer_norm(a)
    b = hidet.ops.softmax(a, axis=1)
    print(a)
    print(b)


def demo_gpu_auto_scheduler():
    import hidet
    a = hidet.randn([3, 4], mean=100, stddev=30, device='cuda')

    @hidet.jit(opt=True)
    def f(x):
        # x1 = x + x
        # x2 = x1 * x
        # x3 = hidet.ops.transpose(x2)
        # return x3
        return hidet.ops.layer_norm(x)

    b = f(a)
    print(a)
    print(b)


if __name__ == '__main__':
    # old()
    # brand_new()
    # new_scheduler_example_2()
    # demo_cpu_auto_scheduler()
    demo_gpu_auto_scheduler()
