import click
from tabulate import tabulate
from hidet.cli.bench.model import BenchModel, all_registered_models


@click.command(name='all')
def bench_all():
    header = BenchModel.headers()
    result = [bench_model.benchmark() for bench_model in all_registered_models]

    click.echo(tabulate(result, headers=header, tablefmt='github', floatfmt='.3f', numalign='right', stralign='left'))
    click.echo('(PyTorch backend: allow_tf32={})'.format(BenchModel.allow_tf32))
