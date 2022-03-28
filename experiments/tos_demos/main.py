from time import time
from hidet import tos
from hidet.tos.tensor import empty
from hidet.tos.models import resnet
from hidet.tos import nn
from hidet.utils import Timer, cuda


def demo_relu():
    x = empty([1, 3, 1, 1], dtype='float32', device='cuda')
    relu = nn.Relu()
    for i in range(2):
        with Timer(msg=f'relu {i}'):
            cuda.device_synchronize()
            y = relu(x)
            cuda.device_synchronize()

    # print(relu)
    # print(x)
    # print(y)


def demo_bn():
    x = empty([1, 3, 1, 1], dtype='float32', name='x', init_method='rand')
    bn = nn.BatchNorm2d(num_features=3)
    y = bn(x)
    print(bn)
    print(x)
    print(bn.running_mean)
    print(bn.running_var)
    print(y)


def demo_basic_block():
    x = Tensor([1, 3, 64, 64], dtype='float32', name='x', init_method='rand')
    block = resnet.BasicBlock(in_channels=3, channels=6)
    y = block(x)
    print(block)


def demo_bottleneck():
    x = Tensor([1, 3, 64, 64], dtype='float32', name='x', init_method='rand')
    block = resnet.Bottleneck(in_channels=3, channels=6, stride=1)
    y = block(x)
    print(block)


def demo_resnet50():
    model = resnet.resnet50()
    x = Tensor([1, 3, 224, 224], dtype='float32', name='x', init_method='rand')
    for t in range(2):
        cuda.device_synchronize()
        with Timer(f'resnet50 {t}'):
            y = model(x)
            cuda.device_synchronize()
    # print(model)


if __name__ == '__main__':
    # demo_bn()
    demo_relu()
    # demo_basic_block()
    # demo_bottleneck()
    # demo_resnet50()
    # from hidet.ffi.cuda import malloc_async, free_async
    # a = malloc_async(1000)
    # b = malloc_async(10)
    # free_async(a)
    # free_async(b)
