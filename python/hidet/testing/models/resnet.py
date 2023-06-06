# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Type, Union, List
import hidet
from hidet.graph import nn


def conv1x1(in_channels, out_channels, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)


def conv3x3(in_channels, out_channels, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.Relu()
        if in_channels != channels * self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_channels=in_channels, out_channels=channels * self.expansion, stride=stride),
                nn.BatchNorm2d(channels * self.expansion),
            )
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        out = self.relu(out + self.downsample(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        expansion = 4
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, channels * expansion)
        self.bn3 = nn.BatchNorm2d(channels * expansion)
        self.relu = nn.Relu()
        if in_channels != channels * expansion or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_channels=in_channels, out_channels=channels * self.expansion, stride=stride),
                nn.BatchNorm2d(channels * self.expansion),
            )
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)) + self.downsample(x))
        return out


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.Relu()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], channels: int, blocks: int, stride: int = 1):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(block(self.in_channels, channels, stride))
                self.in_channels = channels * block.expansion
            else:
                layers.append(block(self.in_channels, channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.max_pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.squeeze(dims=(2, 3))
        x = self.fc(x)
        return x


def load_pretrained_weights(name: str, hidet_module: nn.Module):
    import torch

    torch_resnet: torch.nn.Module = torch.hub.load('pytorch/vision:v0.6.0', name, pretrained=True).eval()

    torch_state_dict = torch_resnet.state_dict()
    hidet_state_dict = {k: hidet.from_torch(v) for k, v in torch_state_dict.items()}

    for p_name, _ in hidet_module.named_parameters():
        if p_name not in hidet_state_dict:
            raise ValueError(f'Parameter {p_name} not found in state dict')

    hidet_module.load_state_dict(hidet_state_dict)
    return hidet_module


def resnet(name: str, layers, block) -> ResNet:
    module = ResNet(block=block, layers=layers)
    load_pretrained_weights(name, module)
    return module


def resnet18() -> ResNet:
    return resnet('resnet18', [2, 2, 2, 2], BasicBlock)


def resnet34() -> ResNet:
    return resnet('resnet34', [3, 4, 6, 3], BasicBlock)


def resnet50() -> ResNet:
    return resnet('resnet50', [3, 4, 6, 3], Bottleneck)


def resnet101() -> ResNet:
    return resnet('resnet101', [3, 4, 23, 3], Bottleneck)


def resnet152() -> ResNet:
    return resnet('resnet152', [3, 8, 36, 3], Bottleneck)
