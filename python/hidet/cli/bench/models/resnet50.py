import torch
import click
from tabulate import tabulate
from .model import BenchModel, all_registered_models, commonly_used_models


class ResNet(BenchModel):
    def __init__(self, batch_size, channels: int, height: int, width: int, model_name: str):
        self.model_name = model_name
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width

    def __str__(self):
        return self.model_name

    def model(self):
        return torch.hub.load('pytorch/vision:v0.6.0', self.model_name, pretrained=True, verbose=False)

    def example_inputs(self):
        args = (torch.randn(self.batch_size, self.channels, self.height, self.width),)
        kwargs = {}
        return args, kwargs


class ResNet18(ResNet):
    def __init__(self, batch_size, channels: int, height: int, width: int):
        super().__init__(batch_size, channels, height, width, 'resnet18')


class ResNet34(ResNet):
    def __init__(self, batch_size, channels: int, height: int, width: int):
        super().__init__(batch_size, channels, height, width, 'resnet34')


class ResNet50(ResNet):
    def __init__(self, batch_size, channels: int, height: int, width: int):
        super().__init__(batch_size, channels, height, width, 'resnet50')


class ResNet101(ResNet):
    def __init__(self, batch_size, channels: int, height: int, width: int):
        super().__init__(batch_size, channels, height, width, 'resnet101')


class ResNet152(ResNet):
    def __init__(self, batch_size, channels: int, height: int, width: int):
        super().__init__(batch_size, channels, height, width, 'resnet152')


resnet_models = {
    'resnet18': ResNet18(1, 3, 224, 224),
    'resnet34': ResNet34(1, 3, 224, 224),
    'resnet50': ResNet50(1, 3, 224, 224),
    'resnet101': ResNet101(1, 3, 224, 224),
    'resnet152': ResNet152(1, 3, 224, 224),
}


all_registered_models.extend(resnet_models.values())
commonly_used_models.append(resnet_models['resnet50'])


@click.command(name='resnet')
@click.option('-n', '--batch-size', default=1, show_default=True, help='Batch size')
@click.option('-c', '--channels', default=3, show_default=True, help='Input channels')
@click.option('-h', '--height', default=224, show_default=True, help='Input image height')
@click.option('-w', '--width', default=224, show_default=True, help='Input image width')
def bench_resnet(batch_size: int, channels: int, height: int, width: int):
    models = {
        'resnet18': ResNet18(batch_size, channels, height, width),
        'resnet34': ResNet34(batch_size, channels, height, width),
        'resnet50': ResNet50(batch_size, channels, height, width),
        'resnet101': ResNet101(batch_size, channels, height, width),
        'resnet152': ResNet152(batch_size, channels, height, width),
    }

    header = BenchModel.headers()
    result = [model.benchmark() for model in models.values()]

    click.echo(tabulate(result, headers=header, tablefmt='github', floatfmt='.3f', numalign='right', stralign='left'))
