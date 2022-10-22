import click
import hidet

from hidet_cli.bench import models


@click.group(name='bench', help='Benchmark models.')
@click.option('--space', default='2', show_default=True, type=click.Choice(['0', '1', '2']),
              help='The search space of the tunable operator')
@click.option('--no-opt', is_flag=True, default=False, show_default=True,
              help='Optimize the model')
@click.option('--cuda-graph', is_flag=True, default=True, show_default=True,
              help='Whether to use CUDA Graph to dispatch the workload')
@click.pass_context
def bench_group(ctx: click.Context, space: str, cuda_graph: bool, no_opt: bool):
    ctx.ensure_object(dict)
    ctx.obj['space'] = int(space)
    ctx.obj['opt'] = not no_opt
    ctx.obj['cuda-graph'] = cuda_graph


for command in [
    models.bench_resnet50
]:
    assert isinstance(command, click.Command)
    bench_group.add_command(command)

