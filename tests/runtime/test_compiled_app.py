import pytest
import hidet
from hidet.testing.models import resnet18
from hidet.runtime import save_compiled_app, load_compiled_app, create_compiled_app


def test_compiled_app():
    module_1 = resnet18().cuda()
    module_2 = resnet18().cuda()

    x1 = hidet.symbol(['batch_size', 3, 224, 224], dtype='float32', device='cuda:0')
    x2 = hidet.symbol([1, 3, 224, 224], dtype='float32', device='cuda:0')

    y1 = module_1(x1)
    y2 = module_2(x2)

    cgraph_1 = hidet.trace_from(y1, inputs=[x1]).build()
    cgraph_2 = hidet.trace_from(y2, inputs=[x2]).build()

    app = create_compiled_app(
        graphs={'graph_1': cgraph_1, 'graph_2': cgraph_2}, modules={}, tensors={}, attributes={}, name='demo_app'
    )

    save_compiled_app(app, 'app.hidet')

    app = load_compiled_app('app.hidet')

    x = hidet.randn([1, 3, 224, 224], device='cuda')
    y1 = app.graphs['graph_1'](x)
    y2 = app.graphs['graph_2'](x)
    hidet.utils.assert_close(y1, y2)

    # check if they share the weights
    # this is one important feature of compiled app that share the weights of graphs if they are numerically identical
    assert len(set(app.graphs['graph_1'].weights) ^ set(app.graphs['graph_2'].weights)) == 0


if __name__ == '__main__':
    pytest.main([__file__])
