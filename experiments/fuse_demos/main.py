import hidet
from hidet.tos import ops
from hidet.backend import build
from hidet.tos.tensor import randn, empty, ones, symbol, randn_like
from hidet.tos.transforms.instruments import SaveGraphInstrument


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
        graph = hidet.tos.optimize(graph)
    graph(randn_like(x))
    # with open('./outs/graph.json', 'w') as f:
    #     hidet.utils.netron.dump(graph, f)


if __name__ == '__main__':
    fuse_conv2d_demo()
