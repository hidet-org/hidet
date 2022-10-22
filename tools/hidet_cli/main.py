from typing import List
import click
from hidet_cli.bench import bench_group


@click.group(name='hidet')
def cli():
    pass


cli.add_command(bench_group)


if __name__ == '__main__':
    cli()
