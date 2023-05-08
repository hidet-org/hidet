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
from hidet.testing.torch_utils import check_module, FunctionalModule


@pytest.mark.parametrize('shape', [[], [2], [2, 3], [2, 3, 4]])
def test_maximum(shape):
    check_module(
        FunctionalModule(op=lambda x, y: torch.maximum(x, y)),
        args=[torch.randn(shape), torch.randn(shape)],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize('shape', [[], [2], [2, 3], [2, 3, 4]])
def test_minimum(shape):
    check_module(
        FunctionalModule(op=lambda x, y: torch.minimum(x, y)),
        args=[torch.randn(shape), torch.randn(shape)],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize('shape', [[2], [2, 3], [2, 3, 4]])
def test_max(shape):
    check_module(FunctionalModule(op=lambda x: torch.max(x)), args=[torch.randn(shape)], atol=1e-5, rtol=1e-5)
    check_module(
        FunctionalModule(op=lambda x, y: torch.max(x, y)),
        args=[torch.randn(shape), torch.randn(shape)],
        atol=1e-5,
        rtol=1e-5,
    )
    check_module(
        FunctionalModule(op=lambda x, dim: torch.max(x, dim)), args=[torch.randn(shape), 0], atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize('shape', [[2], [2, 3], [2, 3, 4]])
def test_min(shape):
    check_module(FunctionalModule(op=lambda x: torch.min(x)), args=[torch.randn(shape)], atol=1e-5, rtol=1e-5)
    check_module(
        FunctionalModule(op=lambda x, y: torch.min(x, y)),
        args=[torch.randn(shape), torch.randn(shape)],
        atol=1e-5,
        rtol=1e-5,
    )
    check_module(
        FunctionalModule(op=lambda x, dim: torch.min(x, dim)), args=[torch.randn(shape), 0], atol=1e-5, rtol=1e-5
    )
