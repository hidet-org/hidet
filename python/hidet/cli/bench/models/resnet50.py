import torch
import click
from tabulate import tabulate
from .model import BenchModel, registered_models


class ResNet50(BenchModel):
    def __init__(self, batch_size, channels: int, height: int, width: int):
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width

    def __str__(self):
        return 'resnet50 ({}x{}x{}x{})'.format(self.batch_size, self.channels, self.height, self.width)

    def model(self):
        return torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

    def example_inputs(self):
        args = (torch.randn(self.batch_size, self.channels, self.height, self.width),)
        kwargs = {}
        return args, kwargs


registered_models.append(ResNet50(1, 3, 224, 224))


@click.command(name='resnet50')
@click.option('-n', '--batch-size', default=1, show_default=True, help='Batch size')
@click.option('-c', '--channels', default=3, show_default=True, help='Input channels')
@click.option('-h', '--height', default=224, show_default=True, help='Input image height')
@click.option('-w', '--width', default=224, show_default=True, help='Input image width')
def bench_resnet50(batch_size: int, channels: int, height: int, width: int):
    bench_model = ResNet50(batch_size, channels, height, width)
    header = bench_model.result_header()
    result = bench_model.bench_all()
    click.echo(tabulate(result, headers=header, tablefmt='github', floatfmt='.3f', numalign='right', stralign='left'))
