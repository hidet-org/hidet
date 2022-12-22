import torch
import click
from tabulate import tabulate
from .model import BenchModel, all_registered_models, commonly_used_models


class ResNet(BenchModel):
    def __init__(self, model_name: str, batch_size, channels: int, height: int, width: int):
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


resnet_models = {
    'resnet18': ResNet('resnet18', 1, 3, 224, 224),
    'resnet34': ResNet('resnet34', 1, 3, 224, 224),
    'resnet50': ResNet('resnet50', 1, 3, 224, 224),
    'resnet101': ResNet('resnet101', 1, 3, 224, 224),
    'resnet152': ResNet('resnet152', 1, 3, 224, 224),
}


all_registered_models.extend(resnet_models.values())
commonly_used_models.append(resnet_models['resnet50'])


@click.command(name='resnet')
@click.option(
    '--models',
    type=str,
    default='resnet50',
    show_default=True,
    help='Comma seperated models to benchmark. Available models: {}'.format(', '.join(list(resnet_models.keys()))),
)
@click.option('-n', '--batch-size', default=1, show_default=True, help='Batch size')
@click.option('-c', '--channels', default=3, show_default=True, help='Input channels')
@click.option('-h', '--height', default=224, show_default=True, help='Input image height')
@click.option('-w', '--width', default=224, show_default=True, help='Input image width')
def bench_resnet(models: str, batch_size: int, channels: int, height: int, width: int):
    models = [model.strip() for model in models.split(',')]
    for model in models:
        if model not in resnet_models:
            raise ValueError('Unknown model: {}, candidates: {}'.format(model, list(resnet_models.keys())))

    bench_models = [ResNet(model_name, batch_size, channels, height, width) for model_name in models]
    header = BenchModel.headers()
    result = [bench_model.benchmark() for bench_model in bench_models]

    click.echo(tabulate(result, headers=header, tablefmt='github', floatfmt='.3f', numalign='right', stralign='left'))
