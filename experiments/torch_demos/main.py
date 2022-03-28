import time
import torch
from torch import nn
from torchvision.models.resnet import resnet50


def demo_resnet50():
    model = resnet50()
    x = torch.rand(128, 3, 224, 224)
    t1 = time.time()
    y = model(x)
    t2 = time.time()
    print(t2 - t1)


if __name__ == '__main__':
    demo_resnet50()
