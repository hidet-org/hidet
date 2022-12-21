import click
from . import models
from hidet.utils import initialize


@click.group(name='bench', help='Benchmark models.')
def bench_group():
    pass


@initialize()
def register_commands():
    for command in [
        models.bench_resnet50
    ]:
        assert isinstance(command, click.Command)
        bench_group.add_command(command)
