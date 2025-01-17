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
import pytest
import torch
from pandas.io.stata import value_label_mismatch_doc

from hidet.testing.torch_utils import check_module, FunctionalModule


@pytest.mark.parametrize(
    "a_shape, b_shape, indices",
    [
        [[10, 11, 12, 13], [10, 9, 8], (slice(None), slice(9), slice(8), 0)],
        [[5, 4, 3, 2], [2, 2], (slice(2, 4), slice(2, 4), 0, 1)],
        [[1, 1, 1024], [10], (0, 0, slice(1000, 1010))],
        [[4, 4, 4, 4], [1, 1, 1], (slice(1), 0, slice(1), slice(1))],
        [[4, 4, 4, 4], [2, 3], (3, slice(2), 2, slice(3))],
        [[4, 4, 4, 4], [2, 3], (slice(2), 0, slice(3), 0)],
        [[4, 4, 4, 4], [4, 4], (0, Ellipsis, 0)],
        [[10, 10], [10, 10], (Ellipsis,)],
        [[1, 3, 28, 28, 85], [1, 3, 28, 28, 2], (Ellipsis, slice(2))],
    ],
)
def test_setitem_with_tensor(a_shape, b_shape, indices, device):
    def check_setitem(x, y, indices):
        x[indices] = y
        return x

    check_module(
        FunctionalModule(op=check_setitem),
        args=[torch.randn(a_shape), torch.randn(b_shape), indices],
        atol=0,
        rtol=0,
        device=device,
    )


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.version.hip is None), reason="CUDA is not available")
@pytest.mark.parametrize('x_dtype, setvalue_dtype', [(torch.float32, torch.float16), (torch.float16, torch.float32)])
@pytest.mark.parametrize("x_device, setvalue_device", [['cuda', 'cpu'], ['cpu', 'cuda']])
def test_setitem_device_dtype_special(x_device, setvalue_device, x_dtype, setvalue_dtype):
    x = torch.randn([3, 3, 3], device=x_device, dtype=x_dtype)
    setvalue = torch.randn([3, 3, 2], device=setvalue_device, dtype=setvalue_dtype)

    def setitem_func(x, setvalue):
        x[..., :2] = setvalue

    setitem_func_compiled = torch.compile(setitem_func, backend='hidet')

    x1 = x.clone()
    setitem_func(x, setvalue)
    setitem_func_compiled(x1, setvalue)

    assert x1.dtype == x.dtype, f"dtype mismatch: {x1.dtype} vs {x.dtype}"
    assert x1.device == x.device, f"device mismatch: {x1.device} vs {x.device}"


@pytest.mark.parametrize(
    "a_shape, setvalue, indices",
    [
        [[10, 11, 12, 13], 1.0, (slice(None), slice(9), slice(8), 0)],
        [[5, 4, 3, 2], 1.0, (slice(2, 4), slice(2, 4), 0, 1)],
        [[1, 1, 1024], 1.0, (0, 0, slice(1000, 1010))],
        [[4, 4, 4, 4], 1.0, (slice(1), 0, slice(1), slice(1))],
        [[4, 4, 4, 4], 1.0, (3, slice(2), 2, slice(3))],
        [[4, 4, 4, 4], 1.0, (slice(2), 0, slice(3), 0)],
        [[4, 4, 4, 4], 1.0, (0, Ellipsis, 0)],
        [[10, 10], 1.0, (Ellipsis,)],
        [[1, 3, 28, 28, 85], 1.0, (Ellipsis, slice(2))],
    ],
)
def test_setitem_with_scalar(a_shape, setvalue, indices, device):
    def check_setitem(x, setvalue, indices):
        x[indices] = setvalue
        return x

    check_module(
        FunctionalModule(op=check_setitem),
        args=[torch.randn(a_shape), setvalue, indices],
        atol=0,
        rtol=0,
        device=device,
    )


if __name__ == '__main__':
    pytest.main([__file__])
