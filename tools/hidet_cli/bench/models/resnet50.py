from typing import List
import click
import numpy as np


@click.command(name='resnet50')
@click.option('-n', '--batch-size', default=1, show_default=True, help='Batch size')
@click.option('-c', '--channels', default=3, show_default=True, help='Input channels')
@click.option('-h', '--height', default=224, show_default=True, help='Input image height')
@click.option('-w', '--width', default=224, show_default=True, help='Input image width')
@click.pass_context
def bench_resnet50(ctx: click.Context, batch_size: int, channels: int, height: int, width: int):
    import hidet
    from hidet.testing.hidet_models import resnet50
    from hidet.testing.utils import benchmark_func

    opt = ctx.obj['opt']
    space = ctx.obj['space']
    use_cuda_graph = ctx.obj['cuda-graph']
    model, inputs = resnet50(batch_size, channels, height, width)
    graph = model.flow_graph_for(inputs)
    hidet.option.search_space(space)
    if opt:
        with hidet.graph.PassContext() as ctx:
            graph = hidet.graph.optimize(graph)

    if use_cuda_graph:
        cuda_graph = graph.cuda_graph()
        def run():
            cuda_graph.run()
    else:
        def run():
            graph(inputs)

    latencies: List[float] = benchmark_func(run, warmup=1, number=5, repeat=5)
    head = '{:>10} {:>15} {:>20} {:>10} {:>10} {:>10} {:>10}\n'.format(
        'BatchSize', 'Model', 'Input', 'Opt', 'Space', 'Latency', 'Std'
    )
    summary = '{:>10} {:>15} {:>20} {:>10} {:>10} {:10.3f} {:10.3f}\n'.format(
        batch_size, 'resnet50', '{}x{}x{}x{}'.format(batch_size, channels, height, width), str(opt), space, np.mean(latencies), np.std(latencies),
    )
    click.echo(head + summary)
