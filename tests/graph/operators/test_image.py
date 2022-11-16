from typing import List

import numpy as np
import torch
import torchvision as tv
import pytest

import hidet
from hidet import ops
from hidet.testing import check_binary
from hidet.graph.tensor import array
from hidet.utils.ort_utils import create_ort_session, ort_inference


class TorchResizeModel(torch.nn.Module):
    def __init__(self, size, method):
        super(TorchResizeModel, self).__init__()
        self.transform = tv.transforms.Resize(size, method)

    def forward(self, x):
        x = self.transform(x)
        return x


def ort_resize2d(data: np.ndarray, size: List[int], method: str):
    method_map = {
        'nearest': tv.transforms.InterpolationMode.NEAREST,
        'linear': tv.transforms.InterpolationMode.BILINEAR,
        'cubic': tv.transforms.InterpolationMode.BICUBIC,
    }
    if method not in method_map:
        raise NotImplementedError(method)

    torch_model = TorchResizeModel(size, method_map[method])
    torch_input = torch.from_numpy(data).cuda()
    torch.onnx.export(torch_model, torch_input, "torch_resize.onnx")
    ort_session = create_ort_session("torch_resize.onnx")
    ort_inputs = {'img': hidet.from_torch(torch_input)}
    ort_outputs = ort_inference(ort_session, ort_inputs)
    ort_output = next(iter(ort_outputs.values()))
    return ort_output.numpy()


def torch_resize2d(data: np.ndarray, size: List[int], method: str):
    method_map = {
        'nearest': tv.transforms.InterpolationMode.NEAREST,
        'linear': tv.transforms.InterpolationMode.BILINEAR,
        'cubic': tv.transforms.InterpolationMode.BICUBIC,
    }
    if method not in method_map:
        raise NotImplementedError(method)
    data_torch = torch.from_numpy(data)
    transform = tv.transforms.Resize(size, method_map[method])
    output = transform(data_torch).numpy()
    return output


# In Pytorch, 'linear' and 'cubic' modes use 'half_pixel' coordinate transformation mode,
# while 'nearest' mode uses 'asymmetric' and 'floor'
@pytest.mark.parametrize(
    "n, c, h, w, size, method, coordinate_transformation_mode, rounding_method, roi, cubic_alpha, cubic_exclude, extrapolation_value",
    [
        [1, 1, 32, 32, [50, 60], 'nearest', 'asymmetric', 'floor', [], -0.75, 0, 0.0],  # nearest upsample
        [1, 1, 32, 32, [20, 15], 'nearest', 'asymmetric', 'floor', [], -0.75, 0, 0.0],  # nearest downsample
        [1, 3, 32, 32, [50, 60], 'linear', 'half_pixel', 'floor', [], -0.75, 0, 0.0],  # linear upsample
        [1, 3, 32, 32, [20, 15], 'linear', 'half_pixel', 'floor', [], -0.75, 0, 0.0],  # linear downsample
        [1, 3, 32, 32, [50, 60], 'cubic', 'half_pixel', 'floor', [], -0.75, 0, 0.0],  # cubic upsample
        [1, 3, 32, 32, [20, 15], 'cubic', 'half_pixel', 'floor', [], -0.75, 0, 0.0],  # cubic downsample
    ],
)
def test_resize2d(
    n,
    c,
    h,
    w,
    size,
    method,
    coordinate_transformation_mode,
    rounding_method,
    roi,
    cubic_alpha,
    cubic_exclude,
    extrapolation_value,
):
    data_shape = [n, c, h, w]
    dtype = 'float32'
    data = np.array(np.random.randn(*data_shape)).astype(dtype)
    torch_result = torch_resize2d(data, size, method)

    hidet_result_cuda = (
        ops.resize2d(
            array(data).to(device='cuda'),
            size,
            method,
            coordinate_transformation_mode,
            rounding_method,
            roi,
            cubic_alpha,
            cubic_exclude,
            extrapolation_value,
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(actual=hidet_result_cuda, desired=torch_result, atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    pytest.main([__file__])
