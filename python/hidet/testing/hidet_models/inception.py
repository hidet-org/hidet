from typing import Tuple, List, Union
import hidet
from hidet.graph import nn, ops, Tensor

# import torchvision.models.inception

# Acknowledgement: the model definitions are adopted from torchvision.models.inception

Ints = Union[int, List[int], Tuple[int]]


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Ints,
        padding: Ints = 0,
        stride: Ints = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride, groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return ops.relu(x)


class InceptionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.max_pool_1(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.max_pool_2(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_2(self.branch5x5_1(x))
        branch3x3 = self.branch3x3_3(self.branch3x3_2(self.branch3x3_1(x)))
        branch_pool = self.branch_pool(ops.avg_pool2d(x, kernel=3, stride=1, padding=1))
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return ops.concat(outputs, axis=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3_p1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_p2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_p3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3_p = self.branch3x3_p1(x)
        branch3x3_p = self.branch3x3_p2(branch3x3_p)
        branch3x3_p = self.branch3x3_p3(branch3x3_p)

        branch_pool = ops.max_pool2d(x, kernel=3, stride=2, padding=0)

        return ops.concat([branch3x3, branch3x3_p, branch_pool], axis=1)


class InceptionC(nn.Module):
    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=[1, 7], padding=[0, 3])
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=[7, 1], padding=[3, 0])

        self.branch7x7_dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_dbl_2 = BasicConv2d(c7, c7, kernel_size=[7, 1], padding=[3, 0])
        self.branch7x7_dbl_3 = BasicConv2d(c7, c7, kernel_size=[1, 7], padding=[0, 3])
        self.branch7x7_dbl_4 = BasicConv2d(c7, c7, kernel_size=[7, 1], padding=[3, 0])
        self.branch7x7_dbl_5 = BasicConv2d(c7, 192, kernel_size=[1, 7], padding=[0, 3])

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7_dbl = self.branch7x7_dbl_1(x)
        branch7x7_dbl = self.branch7x7_dbl_2(branch7x7_dbl)
        branch7x7_dbl = self.branch7x7_dbl_3(branch7x7_dbl)
        branch7x7_dbl = self.branch7x7_dbl_4(branch7x7_dbl)
        branch7x7_dbl = self.branch7x7_dbl_5(branch7x7_dbl)

        branch_pool = ops.avg_pool2d(x, kernel=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return ops.concat([branch1x1, branch7x7, branch7x7_dbl, branch_pool], axis=1)


class InceptionD(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=[1, 7], padding=[0, 3])
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=[7, 1], padding=[3, 0])
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = ops.max_pool2d(x, kernel=3, stride=2, padding=0)
        return ops.concat([branch3x3, branch7x7x3, branch_pool], axis=1)


class InceptionE(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=[1, 3], padding=[0, 1])
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=[3, 1], padding=[1, 0])

        self.branch3x3_dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3_dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3_dbl_3a = BasicConv2d(384, 384, kernel_size=[1, 3], padding=[0, 1])
        self.branch3x3_dbl_3b = BasicConv2d(384, 384, kernel_size=[3, 1], padding=[1, 0])

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = ops.concat(branch3x3, axis=1)

        branch3x3_dbl = self.branch3x3_dbl_1(x)
        branch3x3_dbl = self.branch3x3_dbl_2(branch3x3_dbl)
        branch3x3_dbl = [self.branch3x3_dbl_3a(branch3x3_dbl), self.branch3x3_dbl_3b(branch3x3_dbl)]
        branch3x3_dbl = ops.concat(branch3x3_dbl, axis=1)

        branch_pool = ops.avg_pool2d(x, kernel=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return ops.concat([branch1x1, branch3x3, branch3x3_dbl, branch_pool], axis=1)


class InceptionTail(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: Tensor):
        x = ops.avg_pool2d(x, kernel=x.shape[2:], stride=1, padding=0)
        x = ops.squeeze(x, dims=[2, 3])
        x = self.fc(x)
        return x


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            InceptionHead(),
            InceptionA(in_channels=192, pool_features=32),
            InceptionA(in_channels=256, pool_features=64),
            InceptionA(in_channels=288, pool_features=64),
            InceptionB(in_channels=288),
            InceptionC(in_channels=768, channels_7x7=128),
            InceptionC(in_channels=768, channels_7x7=160),
            InceptionC(in_channels=768, channels_7x7=160),
            InceptionC(in_channels=768, channels_7x7=192),
            InceptionD(in_channels=768),
            InceptionE(in_channels=1280),
            InceptionE(in_channels=2048),
            InceptionTail(),
        )

    def forward(self, x):
        return self.blocks(x)


def basic_conv2d(
    batch_size, in_channels, height, width, out_channels, kernel_size, padding, stride, groups
) -> Tuple[nn.Module, List[Tensor]]:
    inputs = [hidet.randn([batch_size, in_channels, height, width])]
    model = BasicConv2d(in_channels, out_channels, kernel_size, padding, stride, groups)
    return model, inputs


def inception_head(batch_size=1):
    inputs = [hidet.randn([batch_size, 3, 299, 299])]
    model = InceptionHead()
    return model, inputs


def inception_a(in_channels: int = 192, pool_features: int = 32, batch_size=1):
    assert (in_channels, pool_features) in [(192, 32), (256, 64), (288, 64)]
    inputs = [hidet.randn([batch_size, in_channels, 35, 35])]
    model = InceptionA(in_channels, pool_features)
    return model, inputs


def inception_b(batch_size=1):
    inputs = [hidet.randn([batch_size, 288, 35, 35])]
    model = InceptionB(in_channels=288)
    return model, inputs


def inception_c(in_channels=768, channels_7x7=128, batch_size=1):
    assert (in_channels, channels_7x7) in [(768, 128), (768, 160), (768, 160), (768, 192)]
    inputs = [hidet.randn([batch_size, in_channels, 17, 17])]
    model = InceptionC(in_channels, channels_7x7=channels_7x7)
    return model, inputs


def inception_d(in_channels=768, batch_size=1):
    inputs = [hidet.randn([batch_size, in_channels, 17, 17])]
    model = InceptionD(in_channels)
    return model, inputs


def inception_e(in_channels=1280, batch_size=1):
    assert in_channels in [1280, 2048]
    inputs = [hidet.randn([batch_size, in_channels, 8, 8])]
    model = InceptionE(in_channels)
    return model, inputs


def inception_tail(batch_size=1):
    inputs = [hidet.randn([batch_size, 2048, 8, 8])]
    model = InceptionTail()
    return model, inputs


def inception_v3(batch_size=1, channels=3, height=299, width=299):
    inputs = [hidet.randn([batch_size, channels, height, width])]
    model = InceptionV3()
    return model, inputs
