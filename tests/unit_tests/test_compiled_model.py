import pytest
import numpy.testing
import hidet


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_load_save(device: str):
    # construct graph
    x = hidet.symbol([2, 3], device=device)
    w1 = hidet.randn([3, 4], device=device)
    w2 = hidet.randn([4, 5], device=device)
    y = hidet.ops.matmul(hidet.ops.matmul(x, w1), w2)

    # get computation graph
    graph = hidet.trace_from(y)

    # optimize the graph
    graph = hidet.graph.optimize(graph)

    # build the graph
    compiled_graph = graph.build()

    # save the model
    compiled_graph.save('./model.hidet')

    # load the model
    loaded_compiled_graph = hidet.load_compiled_graph('./model.hidet')

    # compare the results
    xx = hidet.randn([2, 3], device=device)
    y1 = graph(xx)
    y2 = compiled_graph(xx)
    y3 = loaded_compiled_graph(xx)

    numpy.testing.assert_allclose(y1.cpu().numpy(), y2.cpu().numpy())
    numpy.testing.assert_allclose(y1.cpu().numpy(), y3.cpu().numpy())
