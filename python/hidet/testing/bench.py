from typing import Tuple, List
from collections import OrderedDict
from hidet.utils import prod


class Conv2dSetting:
    def __init__(self, batch_size, in_channels, image_size, out_channels, kernel, stride, padding):
        image_size, kernel, stride, padding = self.normalize(image_size, kernel, stride, padding)
        self.batch_size: int = batch_size
        self.in_channels: int = in_channels
        self.image_size: Tuple[int, int] = image_size
        self.out_channels: Tuple[int, int] = out_channels
        self.kernel: Tuple[int, int] = kernel
        self.stride: Tuple[int, int] = stride
        self.padding: Tuple[int, int] = padding
        self.output_image_size = tuple([(image_size[i] + 2 * padding[i] - kernel[i]) // stride[i] + 1 for i in range(2)])

    def __str__(self):
        return 'input_{}x{}x{}x{}__kernel_{}x{}_stride_{}x{}_padding_{}x{}_output_{}x{}x{}x{}_flops_{:.0f}'.format(
            self.batch_size, self.in_channels, *self.image_size, *self.kernel, *self.stride,
            *self.padding, self.batch_size, self.out_channels, *self.output_image_size, self.flops()
        )

    def __repr__(self):
        return str(self)

    def flops(self):
        return self.batch_size * self.out_channels * prod(self.output_image_size) * self.in_channels * prod(self.kernel) / 10 ** 6  # M FLOPs

    def keys(self) -> List[str]:
        return ['n', 'ic', 'h', 'w', 'oc', 'kx', 'ky', 'px', 'py', 'sx', 'sy']

    def values(self) -> List[int]:
        return [self.batch_size, self.in_channels, self.image_size[0], self.image_size[1], self.out_channels,
                self.kernel[0], self.kernel[1], self.padding[0], self.padding[1], self.stride[0], self.stride[1]]

    @staticmethod
    def normalize(*args):
        for arg in args:
            if not isinstance(arg, (tuple, list)):
                arg = (arg, arg)
            yield arg

    @staticmethod
    def resnet50_conv2ds(batch_size=1):
        workloads = OrderedDict()
        workloads[Conv2dSetting(batch_size=batch_size, in_channels=3, image_size=224, out_channels=64, kernel=7, stride=2, padding=3)] = 1
        for image_size, channels, repeat in zip([56, 28, 14, 7], [64, 128, 256, 512], [3, 4, 6, 3]):
            if image_size == 56:
                lowering_convs = [
                    (Conv2dSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels, kernel=1, stride=1, padding=0), 1),
                    (Conv2dSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels, kernel=3, stride=1, padding=1), 1),
                    (Conv2dSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels * 4, kernel=1, stride=1, padding=0), 1),
                    (Conv2dSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels * 4, kernel=1, stride=1, padding=0), 1)  # skip connection
                ]
            else:
                lowering_convs = [
                    (Conv2dSetting(batch_size=batch_size, in_channels=channels * 2, image_size=image_size * 2, out_channels=channels, kernel=1, stride=1, padding=0), 1),
                    (Conv2dSetting(batch_size=batch_size, in_channels=channels, image_size=image_size * 2, out_channels=channels, kernel=3, stride=2, padding=1), 1),
                    (Conv2dSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels * 4, kernel=1, stride=1, padding=0), 1),
                    (Conv2dSetting(batch_size=batch_size, in_channels=channels * 2, image_size=image_size * 2, out_channels=channels * 4, kernel=1, stride=2, padding=0), 1)  # skip connection
                ]
            normal_convs = [
                (Conv2dSetting(batch_size=batch_size, in_channels=channels * 4, image_size=image_size, out_channels=channels, kernel=1, stride=1, padding=0), repeat - 1),
                (Conv2dSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels, kernel=3, stride=1, padding=1), repeat - 1),
                (Conv2dSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels * 4, kernel=1, stride=1, padding=0), repeat - 1),
            ]
            for conv, r in lowering_convs + normal_convs:
                if conv not in workloads:
                    workloads[conv] = 0
                workloads[conv] += r
        return workloads

    def __eq__(self, other):
        if len(self.__dict__) != len(other.__dict__):
            return False
        for k in self.__dict__:
            if k not in other.__dict__:
                return False
            if self.__dict__[k] != other.__dict__[k]:
                return False
        return True

    def __hash__(self):
        return hash((self.batch_size, self.in_channels, self.image_size, self.out_channels, self.kernel, self.stride, self.padding))
