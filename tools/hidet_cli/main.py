from typing import List
import click
from hidet_cli.bench import bench_group
from hidet_cli.exec import exec_group
import hidet


@click.group(name='hidet')
def cli():
    pass


for group in [
    bench_group,
    exec_group
]:
    assert isinstance(group, click.Command)
    cli.add_command(group)


if __name__ == '__main__':
    cli()
