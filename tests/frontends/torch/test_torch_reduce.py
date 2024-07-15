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
    check_module(FunctionalModule(op=lambda x: torch.max(x)), args=[torch.randn(shape)], atol=0, rtol=0)
    check_module(
        FunctionalModule(op=lambda x, y: torch.max(x, y)), args=[torch.randn(shape), torch.randn(shape)], atol=0, rtol=0
    )
    check_module(FunctionalModule(op=lambda x, dim: torch.max(x, dim)), args=[torch.randn(shape), 0], atol=0, rtol=0)

    # Do the same checks as above all over again, this time for torch.Tensor.max methods
    check_module(FunctionalModule(op=lambda x: x.max()), args=[torch.randn(shape)], atol=0, rtol=0)
    check_module(FunctionalModule(op=lambda x, dim: x.max(dim)), args=[torch.randn(shape), 0], atol=0, rtol=0)
    check_module(
        FunctionalModule(op=lambda x, y: x.max(y)), args=[torch.randn(shape), torch.randn(shape)], atol=0, rtol=0
    )


@pytest.mark.parametrize('shape', [[2], [2, 3], [2, 3, 4]])
def test_min(shape):
    check_module(FunctionalModule(op=lambda x: torch.min(x)), args=[torch.randn(shape)], atol=0, rtol=0)
    check_module(
        FunctionalModule(op=lambda x, y: torch.min(x, y)),
        args=[torch.randn(shape), torch.randn(shape)],
        atol=1e-5,
        rtol=1e-5,
    )
    check_module(FunctionalModule(op=lambda x, dim: torch.min(x, dim)), args=[torch.randn(shape), 0], atol=0, rtol=0)

    # Doing the same checks as above again, this time for `torch.Tensor.min` method.
    check_module(FunctionalModule(op=lambda x: x.min()), args=[torch.randn(shape)], atol=0, rtol=0)

    check_module(FunctionalModule(op=lambda x, dim: x.min(dim)), args=[torch.randn(shape), 0], atol=0, rtol=0)

    check_module(
        FunctionalModule(op=lambda x, y: x.min(y)), args=[torch.randn(shape), torch.randn(shape)], atol=0, rtol=0
    )


@pytest.mark.parametrize('shape', [[2], [2, 3], [2, 3, 4]])
def test_sum(shape):
    # Similar idea as test_max and test_min
    check_module(FunctionalModule(op=lambda x: torch.sum(x)), args=[torch.randn(shape)], atol=1e-5, rtol=1e-5)

    check_module(
        FunctionalModule(op=lambda x, dim: torch.sum(x, dim)), args=[torch.randn(shape), 0], atol=1e-5, rtol=1e-5
    )

    check_module(FunctionalModule(op=lambda x: x.sum()), args=[torch.randn(shape)], atol=1e-5, rtol=1e-5)

    check_module(FunctionalModule(op=lambda x, dim: x.sum(dim)), args=[torch.randn(shape), 0], atol=1e-5, rtol=1e-5)

    check_module(
        FunctionalModule(op=lambda x, dim: x.sum(dim)),
        args=[torch.randn(shape), list(range(len(shape)))],
        atol=1e-5,
        rtol=1e-5,
    )

    check_module(
        FunctionalModule(op=lambda x, dim: x.sum(dim, keepdim=True)),
        args=[torch.randn(shape), None],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize('shape', [[2], [2, 3], [2, 3, 4]])
def test_mean(shape):
    # Similar idea as test_sum
    check_module(FunctionalModule(op=lambda x: torch.mean(x)), args=[torch.randn(shape)], atol=1e-5, rtol=1e-5)

    check_module(
        FunctionalModule(op=lambda x, dim: torch.mean(x, dim)), args=[torch.randn(shape), 0], atol=1e-5, rtol=1e-5
    )

    check_module(
        FunctionalModule(op=lambda x, dim: torch.mean(x, dim)),
        args=[torch.randn(shape), list(range(len(shape)))],
        atol=1e-5,
        rtol=1e-5,
    )

    check_module(FunctionalModule(op=lambda x: x.mean()), args=[torch.randn(shape)], atol=1e-5, rtol=1e-5)

    check_module(FunctionalModule(op=lambda x, dim: x.mean(dim)), args=[torch.randn(shape), 0], atol=1e-5, rtol=1e-5)

    check_module(
        FunctionalModule(op=lambda x, dim: x.mean(dim)),
        args=[torch.randn(shape), list(range(len(shape)))],
        atol=1e-5,
        rtol=1e-5,
    )
