from hidet import tos
from hidet.tos import Tensor
from hidet.tos.models import resnet
from hidet.tos import nn


def demo_relu():
    x = Tensor([1, 3, 64, 64], dtype='float32', name='x', init_method='rand')
    relu = nn.Relu()
    y = relu(x)
    print(relu)
    print(y)


def demo_bn():
    x = Tensor([1, 3, 64, 64], dtype='float32', name='x', init_method='rand')
    bn = nn.BatchNorm2d(num_features=3)
    y = bn(x)
    print(bn)
    print(y)


def demo_basic_block():
    x = Tensor([1, 3, 64, 64], dtype='float32', name='x', init_method='rand')
    block = resnet.BasicBlock(in_channels=3, channels=6)
    y = block(x)
    print(block)


def demo_resnet50():
    model = resnet.resnet50()
    x = Tensor([1, 3, 224, 224], dtype='float32', name='x', init_method='rand')
    y = model(x)
    print(model)


if __name__ == '__main__':
    demo_bn()
    # demo_relu()
    # demo_basic_block()
    # demo_resnet50()
