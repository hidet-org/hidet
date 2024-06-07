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
import hidet
from hidet.testing.torch_utils import check_module, FunctionalModule


def test_as_torch_tensor():
    """
    test __torch_func__ protocol
    """
    a = hidet.randn([32, 32], dtype='float16', device='cuda')
    b = torch.abs(a)
    c = hidet.ops.abs(a)
    torch.testing.assert_close(b, c.torch())


@pytest.mark.parametrize(
    'shape1,shape2', [([2, 2], [2, 2]), ([2, 3, 4], [2, 3, 4]), ([2, 3, 4], [2, 3, 1]), ([2, 3, 4], [2, 1, 1])]
)
def test_torch_div(shape1, shape2):
    check_module(
        FunctionalModule(op=lambda x, y: torch.div(x, y)),
        args=[torch.randn(shape1), torch.randn(shape2)],
        atol=1e-5,
        rtol=1e-5,
    )

    check_module(
        FunctionalModule(op=lambda x, y: torch.div(x, y, rounding_mode='floor')),
        args=[torch.randn(shape1), torch.randn(shape2)],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize('shape,expanded_shape', [([2, 1], [2, 11]), ([2, 3, 4], [2, 3, 4]), ([1], [6])])
def test_expand_as(shape, expanded_shape):
    check_module(
        FunctionalModule(op=lambda x, y: x.expand_as(y)),
        args=[torch.randn(shape), torch.randn(expanded_shape)],
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize('shape', [[2, 3]])
def test_tensor_sigmod(shape):
    check_module(FunctionalModule(op=lambda x: x.sigmoid_()), args=[torch.randn(shape)], atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
