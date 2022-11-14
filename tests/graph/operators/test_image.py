from typing import List

import numpy as np
import torch
import torchvision as tv
import pytest

from hidet import ops
from hidet.testing import check_binary
from hidet.graph.tensor import array


def torch_resize2d(data: np.ndarray, size: List[int], method: str):
    method_map = {
        'nearest': tv.transforms.InterpolationMode.NEAREST,
        'linear':tv.transforms.InterpolationMode.BILINEAR,
        'cubic': tv.transforms.InterpolationMode.BICUBIC,
    }
    if method not in method_map:
        raise NotImplementedError(method)
    data_torch = torch.from_numpy(data)
    transform = tv.transforms.Resize(size, method_map[method])
    output = transform(data_torch).numpy()
    return output


@pytest.mark.parametrize(
    "n, c, h, w, size, method, coordinate_transformation_mode, rounding_method, roi, cubic_alpha, cubic_exclude, extrapolation_value",
    [
        #[1, 3, 32, 32, [50,60], 'nearest', 'pytorch_half_pixel', 'round_prefer_floor', [], -0.75, 0, 0.0], # nearest upsample
        [1, 3, 32, 32, [50,60], 'linear', 'pytorch_half_pixel', 'round_prefer_floor', [], -0.75, 0, 0.0], # linear upsample
        #[1, 3, 32, 32, [20,15], 'linear', 'pytorch_half_pixel', 'round_prefer_floor', [], -0.75, 0, 0.0], # linear downsample
        #[1, 3, 32, 32, [50,60], 'cubic', 'pytorch_half_pixel', 'round_prefer_floor', [], -0.75, 0, 0.0], # cubic upsample
        #[1, 3, 32, 32, [20,15], 'cubic', 'pytorch_half_pixel', 'round_prefer_floor', [], -0.75, 0, 0.0], # cubic downsample
        #[1, 3, 32, 32, [50,60], 'cubic', 'pytorch_half_pixel', 'round_prefer_floor', [], -0.75, 1, 0.0], # cubic upsample exclude
    ],
)
def test_resize2d(n, c, h, w, size, method, coordinate_transformation_mode, rounding_method, roi, cubic_alpha, cubic_exclude, extrapolation_value):
    data_shape = [n, c, h, w]
    dtype = 'float32'
    data = np.array(np.random.randn(*data_shape)).astype(dtype)
    torch_result = torch_resize2d(data, size, method)
    hidet_result_cpu = ops.resize2d(array(data).to(device='cpu'), size, method, coordinate_transformation_mode, rounding_method, roi, cubic_alpha, cubic_exclude, extrapolation_value).cpu().numpy()
    hidet_result_cuda = ops.resize2d(array(data).to(device='cuda'), size, method, coordinate_transformation_mode, rounding_method, roi, cubic_alpha, cubic_exclude, extrapolation_value).cpu().numpy()
    #np.testing.assert_allclose(actual=hidet_result_cpu, desired=torch_result, atol = 2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual=hidet_result_cuda, desired=torch_result, atol = 2e-5, rtol=2e-5)

if __name__ == '__main__':
    pytest.main([__file__])
