import hidet
import numpy as np
from hidet.tos import ops
from hidet.backend import build
from hidet.tos.tensor import randn, empty, ones, symbol, randn_like
from hidet.tos.transforms.instruments import SaveGraphInstrument, ProfileInstrument


def fuse_conv2d_demo():
    x = symbol([1, 3, 32, 32])
    w = randn([6, 3, 1, 1])
    y = ops.conv2d(x, w, padding=0, stride=1)
    graph = hidet.trace_from(y, [x])
    with hidet.tos.PassContext(
            instruments=[
                SaveGraphInstrument('./outs/graphs')
            ],
            verbose=True
    ):
        opt_graph = hidet.tos.optimize(graph)

    dummy_x = randn_like(x)
    y1 = graph(dummy_x)
    y2 = opt_graph(dummy_x)

    np.testing.assert_allclose(y1.numpy(), y2.numpy())

    # with open('./outs/graph.json', 'w') as f:
    #     hidet.utils.netron.dump(graph, f)


def fuse_resnet50():
    x = symbol([1, 3, 224, 224])
    model = hidet.tos.models.resnet50()
    y = model(x)

    graph = hidet.trace_from(y)

    with hidet.tos.PassContext(
            instruments=[
                SaveGraphInstrument('./outs/resnet50/graphs'),
                ProfileInstrument(log_file='./outs/resnet50/graphs/lower_time.txt')
            ],
            verbose=True
    ):
        opt_graph = hidet.tos.optimize(graph)

    dummy_x = randn_like(x)
    y1 = graph(dummy_x)
    y2 = opt_graph(dummy_x)

    np.testing.assert_allclose(y1.numpy(), y2.numpy())


if __name__ == '__main__':
    # fuse_conv2d_demo()
    fuse_resnet50()
