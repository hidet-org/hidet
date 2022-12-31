import click
from tabulate import tabulate
from hidet.cli.bench.model import BenchModel, commonly_used_models


@click.command(name='common')
def bench_common():
    header = BenchModel.headers()
    result = [bench_model.benchmark() for bench_model in commonly_used_models]

    click.echo(tabulate(result, headers=header, tablefmt='github', floatfmt='.3f', numalign='right', stralign='left'))
    click.echo('(PyTorch backend: allow_tf32={})'.format(BenchModel.allow_tf32))
