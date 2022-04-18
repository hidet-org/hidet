import torch
import time

import hidet
from hidet.ffi import cuda_api
from hidet import tos
from hidet.tos.tensor import empty, randn, symbol
from hidet.tos import Operator, Tensor
from hidet.runtime.storage import cuda_pool, Storage
from hidet.tos.models import resnet
from hidet.tos import nn, optimize
from hidet.utils import Timer, cuda, netron, nvtx_annotate
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


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.const = hidet.empty(shape=[1])

    def forward(self, x):
        return -(x - self.const)


def demo_lazy_mode():
    x = symbol([1, 3, 224, 224], dtype='float32')
    model = resnet.resnet50()
    # model = nn.MaxPool2d(kernel_size=7, stride=2, padding=1)
    # model = nn.Sequential(
    #     nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0, stride=1),
    #     nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0, stride=1),
    #     nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0, stride=1),
    # )
    y = model(x)

    graph: hidet.FlowGraph = hidet.trace_from(y)
    # x = randn([16, 3, 224, 224], dtype='float32')
    # for t in range(10):
    #     cuda.device_synchronize()
    #     with Timer('optimized'):
    #         y = graph(x)
    #         cuda.device_synchronize()
    # y = graph(x)
    with tos.PassContext(instruments=[
        SaveGraphInstrument(out_dir='./outs/'),
        ProfileInstrument(log_file='./outs/profile.txt', print_stdout=True)
    ]):
        graph = optimize(graph)

    hidet.space_level(2)
    x = randn([16, 3, 224, 224], dtype='float32')
    for t in range(10):
        with nvtx_annotate('hidet {}'.format(t)):
            cuda.device_synchronize()
            with Timer('optimized'):
                y = graph(x)
                cuda.device_synchronize()

    print(cuda_pool.active_blocks)
    cuda_pool.clear()


def demo_torch_resnet50():
    from torchvision.models import resnet50
    model = resnet50().cuda()
    model.train(False)
    x = torch.rand(1, 3, 224, 224).cuda()
    for t in range(10):
        with nvtx_annotate('torch resnet50 {}'.format(t)):
            torch.cuda.synchronize()
            with Timer(f'torch {t}'):
                y = model(x)
                torch.cuda.synchronize()


def demo_hidet_resnet50():
    x = symbol([1, 3, 224, 224], dtype='float32')
    model = resnet.resnet50()
    y = model(x)
    graph = hidet.trace_from(y)

    x = randn([1, 3, 224, 224], dtype='float32')
    for t in range(10):
        cuda_api.device_synchronization()
        with Timer('hidet resnet50 {}'.format(t)):
            y = graph(x)
            cuda_api.device_synchronization()


if __name__ == '__main__':
    # demo_relu()
    # demo_bn()
    # demo_basic_block()
    # demo_bottleneck()
    # demo_resnet50()
    demo_lazy_mode()
    # demo_torch_resnet50()
    # demo_hidet_resnet50()
