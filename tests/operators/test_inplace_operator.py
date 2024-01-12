import hidet
from hidet.ir.task import Task
from hidet.ir import primitives
from hidet.ir.dtypes import float32
from hidet.runtime import CompiledGraph
from hidet.graph.ops.utils import input_like, Operator, Tensor, compute, TensorInput


class InplaceReLUTask(Task):
    def __init__(self, x: TensorInput):
        y = compute(name='y', shape=x.shape, fcompute=lambda *indices: primitives.max(x[indices], x.type.dtype.zero))
        super().__init__(name='inplace_relu', inputs=[x], outputs=[y], share_map={0: 0})  # share y with x


class InplaceReLUOp(Operator):
    def __init__(self, x: Tensor):
        super().__init__(inputs=[x], attributes={}, task=InplaceReLUTask(input_like(x, 'x')))


def inplace_relu(x: Tensor) -> Tensor:
    return InplaceReLUOp(x).outputs[0]


def test_inplace_relu():
    x = hidet.symbol([1, 10], dtype=float32, device='cuda')
    y = inplace_relu(x)
    graph = hidet.trace_from(y)
    xx = hidet.randn_like(x)

    y1 = graph(xx)
    assert y1.storage is xx.storage

    compiled_graph: CompiledGraph = graph.build()

    compiled_graph.dispatch_table.clear()
    y2 = compiled_graph(xx)  # run in slow path
    assert y2.storage is xx.storage

    y3 = compiled_graph(xx)  # run in fast path
    assert y3.storage is xx.storage

    # y1, y2, y3 should be equal
    yy = hidet.ops.relu(xx)
    hidet.utils.assert_close(y1, yy)


if __name__ == '__main__':
    test_inplace_relu()
