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
import hidet
import torch


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_real(shape, dtype):
    a = torch.randn(shape, dtype=dtype)
    a_hidet = hidet.from_torch(a)

    assert torch.allclose(hidet.ops.real(a_hidet).torch(), torch.real(a))


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_imag(shape, dtype):
    a = torch.randn(shape, dtype=dtype)
    a_hidet = hidet.from_torch(a)

    assert torch.allclose(hidet.ops.imag(a_hidet).torch(), torch.imag(a))


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_conj(shape, dtype):
    a = torch.randn(shape, dtype=dtype)
    a_hidet = hidet.from_torch(a)

    assert torch.allclose(hidet.ops.conj(a_hidet).torch(), torch.conj(a))


def test_make_complex():
    pass
