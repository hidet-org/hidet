import click
from tabulate import tabulate
from hidet.cli.bench.model import BenchModel, all_registered_models, commonly_used_models
from .vision_model import VisionModel


inception_models = {'inception_v3': VisionModel('inception_v3', 1, 3, 299, 299)}


all_registered_models.extend(inception_models.values())
commonly_used_models.append(inception_models['inception_v3'])


@click.command(name='inception-v3')
@click.option('-n', '--batch-size', default=1, show_default=True, help='Batch size')
@click.option('-c', '--channels', default=3, show_default=True, help='Input channels')
@click.option('-h', '--height', default=224, show_default=True, help='Input image height')
@click.option('-w', '--width', default=224, show_default=True, help='Input image width')
def bench_inception_v3(batch_size: int, channels: int, height: int, width: int):
    bench_models = [VisionModel('inception_v3', batch_size, channels, height, width)]
    header = BenchModel.headers()
    result = [bench_model.benchmark() for bench_model in bench_models]

    click.echo(tabulate(result, headers=header, tablefmt='github', floatfmt='.3f', numalign='right', stralign='left'))
    click.echo('(PyTorch backend: allow_tf32={})'.format(BenchModel.allow_tf32))
