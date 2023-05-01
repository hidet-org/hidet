# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

import numpy as np
import torch
from torch.nn import functional as F
import torchvision as tv
import pytest

import hidet
from hidet import ops
from hidet.testing import check_binary
from hidet.graph.tensor import asarray
from hidet.utils.ort_utils import create_ort_session, ort_inference
from hidet.testing import check_torch_unary
from hidet.graph.frontend.torch import register_functions as regs


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
    "n, c, h, w, size, scale_factor, method, coordinate_transformation_mode, rounding_method, "
    "roi, cubic_alpha, cubic_exclude, extrapolation_value, recompute_scale_factor",
    [
        [1, 1, 32, 32, [50, 60], None, 'nearest', 'asymmetric', 'floor', [], -0.75, 0, 0.0, None],  # nearest upsample
        [1, 1, 32, 32, None, 1.5, 'nearest', 'asymmetric', 'floor', [], -0.75, 0, 0.0, None],  # nearest upsample
        [1, 1, 32, 32, [20, 15], None, 'nearest', 'asymmetric', 'floor', [], -0.75, 0, 0.0, None],  # nearest downsample
        [1, 3, 32, 32, [50, 60], None, 'linear', 'half_pixel', 'floor', [], -0.75, 0, 0.0, None],  # linear upsample
        [1, 3, 32, 32, [20, 15], None, 'linear', 'half_pixel', 'floor', [], -0.75, 0, 0.0, None],  # linear downsample
        [1, 3, 32, 32, [50, 60], None, 'cubic', 'half_pixel', 'floor', [], -0.75, 0, 0.0, None],  # cubic upsample
        [1, 3, 32, 32, [20, 15], None, 'cubic', 'half_pixel', 'floor', [], -0.75, 0, 0.0, None],  # cubic downsample
        [1, 1, 37, 37, [16, 16], None, 'cubic', 'half_pixel', 'floor', [], -0.75, 0, 0.0, None],  # cubic downsample
        [1, 1, 37, 37, None, 0.4781, 'cubic', 'half_pixel', 'floor', [], -0.75, 0, 0.0, None],  # cubic downsample
    ],
)
def test_resize2d(
    n,
    c,
    h,
    w,
    size,
    scale_factor,
    method: str,
    coordinate_transformation_mode: str,
    rounding_method: str,
    roi,
    cubic_alpha: float,
    cubic_exclude: bool,
    extrapolation_value: float,
    recompute_scale_factor: bool,
):
    data_shape = [n, c, h, w]
    dtype = 'float32'
    data = np.array(np.random.randn(*data_shape)).astype(dtype)
    # torch_result = torch_resize2d(data, size, method)

    method_rename = {'nearest': 'nearest', 'linear': 'bilinear', 'cubic': 'bicubic'}
    torch_result = torch.nn.functional.interpolate(
        torch.from_numpy(data), size=size, scale_factor=scale_factor, mode=method_rename[method]
    )

    hidet_result_cuda = (
        ops.resize2d(
            asarray(data).to(device='cuda'),
            size=size,
            scale_factor=scale_factor,
            method=method,
            coordinate_transformation_mode=coordinate_transformation_mode,
            rounding_method=rounding_method,
            roi=roi,
            cubic_alpha=cubic_alpha,
            cubic_exclude=cubic_exclude,
            extrapolation_value=extrapolation_value,
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(actual=hidet_result_cuda, desired=torch_result, atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    pytest.main([__file__])
