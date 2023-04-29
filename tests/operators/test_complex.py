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
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_real(shape, dtype, device):
    a = torch.randn(shape, dtype=dtype, device=device)
    a_hidet = hidet.from_torch(a)

    torch.testing.assert_allclose(hidet.ops.real(a_hidet).torch(), torch.real(a))


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_imag(shape, dtype, device):
    a = torch.randn(shape, dtype=dtype, device=device)
    a_hidet = hidet.from_torch(a)

    torch.testing.assert_allclose(hidet.ops.imag(a_hidet).torch(), torch.imag(a))


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_conj(shape, dtype, device):
    a = torch.randn(shape, dtype=dtype, device=device)
    a_hidet = hidet.from_torch(a)

    torch.testing.assert_allclose(hidet.ops.conj(a_hidet).torch(), torch.conj(a))


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_make_complex(shape, dtype, device):
    real = torch.randn(shape, dtype=dtype, device=device)
    imag = torch.randn(shape, dtype=dtype, device=device)
    a = torch.complex(real, imag)
    b = hidet.ops.make_complex(hidet.from_torch(real), hidet.from_torch(imag))

    torch.testing.assert_allclose(b.torch(), a)


@pytest.mark.parametrize("a_shape,b_shape", [[[1, 33, 44], [1, 44, 55]]])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_complex_matmul(a_shape, b_shape, dtype, device):
    a = torch.randn(a_shape, dtype=dtype, device=device)
    b = torch.randn(b_shape, dtype=dtype, device=device)
    c = torch.matmul(a, b)

    a_hidet = hidet.from_torch(a)
    b_hidet = hidet.from_torch(b)
    c_hidet = hidet.ops.matmul(a_hidet, b_hidet)

    torch.testing.assert_allclose(c_hidet.torch(), c, atol=1e-5, rtol=1e-5)
