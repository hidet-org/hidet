from time import time

import hidet
from hidet import tos
from hidet.tos.tensor import empty, randn, symbol
from hidet.tos.models import resnet
from hidet.tos import nn, ops, optimize
from hidet.utils import Timer, cuda, netron
from hidet.tos.transforms import ProfileInstrument, SaveGraphInstrument


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


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.const = hidet.empty(shape=[1])

    def forward(self, x):
        return -(x - self.const)


def demo_lazy_mode():
    hidet.lazy_mode()
    x = symbol([16, 3, 224, 224], dtype='float32')
    # model = nn.Sequential(
    #     resnet.Bottleneck(in_channels=3, channels=16, stride=2),
    #     resnet.Bottleneck(in_channels=64, channels=16, stride=1)
    # )
    # model = nn.Relu()
    model = resnet.resnet50()
    # model = ExampleModel()
    y = model(x)

    graph: hidet.FlowGraph = hidet.trace_from(y)
    x = randn([16, 3, 224, 224], dtype='float32')
    for t in range(10):
        cuda.device_synchronize()
        with Timer('optimized'):
            y = graph(x)
            cuda.device_synchronize()
    y = graph(x)
    with tos.PassContext(instruments=[
        SaveGraphInstrument(out_dir='./outs/'),
        ProfileInstrument(log_file='./outs/profile.txt', print_stdout=True)
    ]):
        graph = optimize(graph)

    y = graph(x)
    for t in range(10):
        cuda.device_synchronize()
        with Timer('optimized'):
            y = graph(x)
            cuda.device_synchronize()


if __name__ == '__main__':
    # demo_relu()
    # demo_bn()
    # demo_basic_block()
    # demo_bottleneck()
    # demo_resnet50()
    demo_lazy_mode()
    pass
