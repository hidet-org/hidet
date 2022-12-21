import click
from hidet.utils import initialize
from . import models


@click.group(name='bench', help='Benchmark models.')
def bench_group():
    pass


@initialize()
def register_commands():
    for command in [models.bench_resnet]:
        assert isinstance(command, click.Command)
        bench_group.add_command(command)
