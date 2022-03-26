from typing import Type, Union, List, Callable, Any
from hidet.tos import nn


def conv1x1(in_channels, out_channels, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)


def conv3x3(in_channels, out_channels, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 skip: Callable[[Any], Any] = None
                 ):
        super().__init__()
        if skip is None:
            skip = (lambda v: v)
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.Relu()
        self.skip = skip if skip is not None else (lambda x: x)

    def forward(self, x):
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        out = self.relu(out + self.skip(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int,
                 skip: Callable[[Any], Any]
                 ):
        super().__init__()
        expansion = 4
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, channels * expansion)
        self.bn3 = nn.BatchNorm2d(channels * expansion)
        self.relu = nn.Relu()
        self.skip = skip

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)) + self.skip(x))
        return out


class ResNet(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.Relu()
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self,
                   block: Type[Union[BasicBlock, Bottleneck]],
                   channels: int,
                   blocks: int,
                   stride: int = 1):
        if stride != 1 or self.in_channels != channels * block.expansion:
            skip = nn.Sequential(
                conv1x1(in_channels=self.in_channels, out_channels=channels * block.expansion, stride=stride),
                nn.BatchNorm2d(channels * block.expansion)
            )
        else:
            skip = (lambda x: x)

        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(block(self.in_channels, channels, stride, skip))
                self.in_channels = channels * block.expansion
            else:
                layers.append(block(self.in_channels, channels, stride=1, skip=lambda v: v))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.max_pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


def resnet18():
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2])


def resnet34():
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3])


def resnet50():
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3])


def resnet101():
    return ResNet(block=Bottleneck, layers=[3, 4, 23, 3])


def resnet152():
    return ResNet(block=Bottleneck, layers=[3, 8, 36, 3])
