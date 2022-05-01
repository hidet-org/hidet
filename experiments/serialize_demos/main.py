import os
import numpy as np
import hidet


def demo_task():
    a = hidet.symbol([1, 1024, 1024])
    b = hidet.symbol([1, 1024, 1024])
    c = hidet.ops.batched_matmul(a, b)
    task = c.op.task
    print(task)
    hidet.save_task(task, './outs/matmul_task.pickle')
    task = hidet.load_task('./outs/matmul_task.pickle')
    print(task)
    task.implement('cuda')


def demo_graph():
    model = hidet.tos.models.resnet50()
    x = hidet.symbol([1, 3, 224, 224])
    y = model(x)
    graph = hidet.trace_from(y)

    dummy_x = hidet.randn_like(x)
    y1 = graph(dummy_x)

    hidet.save_graph(graph, './outs/graph.pickle')
    graph = hidet.load_graph('./outs/graph.pickle')

    y2 = graph(dummy_x)

    np.testing.assert_allclose(y1.numpy(), y2.numpy())


def demo_load_graph():
    graph = hidet.load_graph(hidet.utils.hidet_cache_file('hidet_graph', 'resnet50', 'bs_1', 'graph.pickle'))
    with open('./outs/graph.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)


if __name__ == '__main__':
    # demo_task()
    # demo_graph()
    demo_load_graph()
