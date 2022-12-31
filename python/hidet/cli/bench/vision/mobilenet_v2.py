import click
from tabulate import tabulate
from hidet.cli.bench.model import BenchModel, all_registered_models, commonly_used_models
from .vision_model import VisionModel


inception_models = {'mobilenet_v2': VisionModel('mobilenet_v2', 1, 3, 224, 224)}


all_registered_models.extend(inception_models.values())
commonly_used_models.append(inception_models['mobilenet_v2'])


@click.command(name='mobilenet-v2')
@click.option('-n', '--batch-size', default=1, show_default=True, help='Batch size')
@click.option('-c', '--channels', default=3, show_default=True, help='Input channels')
@click.option('-h', '--height', default=224, show_default=True, help='Input image height')
@click.option('-w', '--width', default=224, show_default=True, help='Input image width')
def bench_mobilenet_v2(batch_size: int, channels: int, height: int, width: int):
    bench_models = [VisionModel('mobilenet_v2', batch_size, channels, height, width)]
    header = BenchModel.headers()
    result = [bench_model.benchmark() for bench_model in bench_models]

    click.echo(tabulate(result, headers=header, tablefmt='github', floatfmt='.3f', numalign='right', stralign='left'))
    click.echo('(PyTorch backend: allow_tf32={})'.format(BenchModel.allow_tf32))
