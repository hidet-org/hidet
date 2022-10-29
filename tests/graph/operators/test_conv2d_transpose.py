import pytest
import numpy as np
import torch
import hidet


@pytest.mark.parametrize(
    'in_channels, out_channels, kernel_size, stride, pads, groups, height, width, output_padding',
    [[10, 20, (5, 5), (3, 2), [2, 1], 5, 11, 10, (2, 1)]],
)
def test_conv2d_transpose(in_channels, out_channels, kernel_size, stride, pads, groups, height, width, output_padding):
    torch_data = torch.ones(1, in_channels, height, width, dtype=torch.float32).cuda()
    torch_weight = torch.ones(
        out_channels, in_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.float32
    ).cuda()

    torch_output = torch.nn.functional.conv2d(
        torch_data, torch_weight, stride=stride, padding=pads, groups=groups, bias=None, dilation=1
    )
    hidet_data = hidet.from_torch(torch_data)
    hidet_weight = hidet.from_torch(torch_weight)
    hidet_output = hidet.ops.conv_pad(hidet_data, pads)
    hidet_output = hidet.ops.conv2d(hidet_output, hidet_weight, stride, groups)
    np.testing.assert_allclose(hidet_output.numpy(), torch_output.cpu().numpy(), atol=1e-5)
    torch_transpose_output = torch.nn.functional.conv_transpose2d(
        torch_output,
        torch_weight,
        stride=stride,
        padding=pads,
        groups=groups,
        bias=None,
        dilation=1,
        output_padding=output_padding,
    )
    hidet_transpose_output = hidet.ops.conv2d_transpose(
        hidet_output, hidet_weight, stride, pads, groups, output_padding=output_padding
    )
    np.testing.assert_allclose(hidet_transpose_output.numpy(), torch_transpose_output.cpu().numpy(), atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
