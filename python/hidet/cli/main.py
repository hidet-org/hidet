import click
from hidet.cli.bench import bench_group
from hidet.utils import initialize


@click.group(name='hidet')
def main():
    pass


@initialize()
def register_commands():
    for group in [bench_group]:
        assert isinstance(group, click.Command)
        main.add_command(group)


if __name__ == '__main__':
    main()
