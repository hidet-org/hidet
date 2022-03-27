import torch
from torch import nn
from torchvision.models.resnet import resnet50

if __name__ == '__main__':
    model = resnet50()
    print(model)
    nn.Module
