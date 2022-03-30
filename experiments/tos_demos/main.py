from time import time
from hidet import tos
from hidet.tos.tensor import empty, randn
from hidet.tos.models import resnet
from hidet.tos import nn, ops
from hidet.utils import Timer, cuda


def demo_relu():
    x = randn([1, 8, 1, 1], dtype='float32', device='cuda')
    relu = nn.Relu()
    for i in range(2):
        cuda.device_synchronize()
        with Timer(msg=f'relu {i}'):
            y = relu(x)
            cuda.device_synchronize()
    print(x)
    print(y)
    print(relu)


def demo_bn():
    x = randn([1, 3, 64, 64], dtype='float32')
    bn = nn.BatchNorm2d(num_features=3)
    for i in range(1):
        with Timer('bn {}'.format(i)):
            y = bn(x)
    print(bn)


def demo_basic_block():
    x = randn([1, 3, 128, 128], dtype='float32')
    block = resnet.BasicBlock(in_channels=3, channels=6)
    for i in range(3):
        cuda.device_synchronize()
        with Timer('basic_block {}'.format(i)):
            y = block(x)
            cuda.device_synchronize()
    print(block)


def demo_bottleneck():
    x = randn([1, 3, 64, 64], dtype='float32')
    block = resnet.Bottleneck(in_channels=3, channels=6, stride=1)
    y = block(x)
    print(block)


def demo_resnet50():
    model = resnet.resnet50()
    x = randn([32, 3, 224, 224], dtype='float32')
    for t in range(10):
        cuda.device_synchronize()
        with Timer(f'resnet50 {t}'):
            y = model(x)
            cuda.device_synchronize()
    # print(model)


if __name__ == '__main__':
    # demo_relu()
    # demo_bn()
    # demo_basic_block()
    # demo_bottleneck()
    demo_resnet50()
    # from hidet.ffi.cuda import malloc_async, free_async
    # a = malloc_async(1000)
    # b = malloc_async(10)
    # free_async(a)
    # free_async(b)
    pass
