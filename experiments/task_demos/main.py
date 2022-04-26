import hidet
from hidet.tos.task import Task
from hidet.tos import ops
from hidet.tos.models import resnet50
from hidet.utils import Timer
from hidet.ffi import cuda_api


def demo_latency():
    model = resnet50()
    x = hidet.randn([1, 3, 224, 224])
    for i in range(10):
        cuda_api.device_synchronization()
        with Timer('{}'.format(i)):
            y = model(x)
            cuda_api.device_synchronization()


def demo_task():
    x = hidet.randn(shape=[3, 4])
    y = hidet.randn(shape=[2, 4])
    # y = hidet.tos.ops.softmax(x, axis=1)
    z = hidet.ops.concat([x, y], axis=0)
    print(x)
    print(y)
    print(z)


if __name__ == '__main__':
    demo_task()
