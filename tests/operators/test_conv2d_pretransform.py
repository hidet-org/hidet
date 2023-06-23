import torch
import hidet
import pytest

from hidet.graph.ops.conv2d.conv2d_gemm import pre_transform_img, pre_transform_imgv2, pre_transform_img
from hidet.graph.ops.conv2d.conv2d_gemm import Conv2dGemmFp16PretransformTask
from hidet.graph.ops.utils import Operator, input_like


# @pytest.mark.parametrize("img_dim", [[32, 64], [31, 63]])
# @pytest.mark.parametrize("channel", [3, 32, 64])
# @pytest.mark.parametrize("padding", [[0, 0], [1, 1], [2, 3]])
# @pytest.mark.parametrize("multi_8", [True, False])
# def test_pretransform_v2(img_dim, channel, padding, multi_8):
#     img = hidet.randn([1, channel] + img_dim, device='cuda', dtype='float16')
#     y1 = pre_transform_img(img, tuple(padding), 0.0, multi_8)
#     y2 = pre_transform_imgv2(img, tuple(padding), 0.0, multi_8)
#     assert torch.allclose(y1.torch(), y2.torch(), 1e-3, 1e-3)


@pytest.mark.parametrize("img_dim", [[32, 64], [31, 63]])
@pytest.mark.parametrize("channel", [3, 32, 64])
@pytest.mark.parametrize("padding", [[0, 0], [1, 1], [2, 3]])
@pytest.mark.parametrize("multi_8", [True, False])
def test_pretransform_v3(img_dim, channel, padding, multi_8):
    img = hidet.randn([1, channel] + img_dim, device='cuda', dtype='float16')
    y1 = pre_transform_img(img, tuple(padding), 0.0, multi_8)
    y2 = pre_transform_img(img, tuple(padding), 0.0, multi_8)
    assert torch.allclose(y1.torch(), y2.torch(), 1e-3, 1e-3)

    imgs = hidet.symbol([1, channel] + img_dim, dtype='float16', device='cuda')
    ys = pre_transform_img(imgs, tuple(padding),0.0, multi_8)
    graph = hidet.trace_from(ys, imgs)
    cgraph = graph.build(space=2)
    task = cgraph.compiled_tasks[0]
    for func in task.candidates:
        y2 = hidet.empty_like(y1)
        func(img, y2)
        assert torch.allclose(y1.torch(), y2.torch(), 1e-3, 1e-3)


if __name__ == '__main__':
    pytest.main([__file__])
