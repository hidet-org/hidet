from typing import Union, Tuple

import torch
import hidet
import pytest

from hidet import Tensor
from hidet.graph.ops.conv2d.conv2d_gemm import pre_transform_img


def pre_transform_img_ref(img: Tensor, padding: Union[int, Tuple[int, int]], pad_value=0.0, make_multiple_8=False):
    import hidet

    n, c, w, h = img.shape
    assert pad_value == 0.0
    img = hidet.ops.conv_pad(img, padding)
    img = hidet.ops.transpose(img, [0, 2, 3, 1])
    if make_multiple_8:
        pad_channel = ((c + 7) // 8) * 8 - c
        img = hidet.ops.pad(img, [0, pad_channel])
    return img


@pytest.mark.parametrize("img_dim", [[32, 64], [31, 63]])
@pytest.mark.parametrize("channel", [3, 32, 64])
@pytest.mark.parametrize("padding", [[0, 0], [1, 1], [2, 3]])
@pytest.mark.parametrize("multi_8", [True, False])
def test_pretransform_v3(img_dim, channel, padding, multi_8):
    img = hidet.randn([1, channel] + img_dim, device='cuda', dtype='float16')
    y1 = pre_transform_img_ref(img, tuple(padding), 0.0, multi_8)
    y2 = pre_transform_img(img, tuple(padding), 0.0, multi_8)
    assert torch.allclose(y1.torch(), y2.torch(), 1e-3, 1e-3)

    imgs = hidet.symbol([1, channel] + img_dim, dtype='float16', device='cuda')
    ys = pre_transform_img(imgs, tuple(padding), 0.0, multi_8)
    graph = hidet.trace_from(ys, imgs)
    cgraph = graph.build(space=2)
    task = cgraph.compiled_tasks[0]
    for func in task.candidates:
        y2 = hidet.empty_like(y1)
        func(img, y2)
        assert torch.allclose(y1.torch(), y2.torch(), 1e-2, 1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
