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
import numpy as np
import pytest
import numpy as np
import torch
import hidet
from hidet.testing import device_to_torch


def test_from_dlpack(device):
    torch_device = device_to_torch(device)

    a = torch.randn([2, 3])
    b = hidet.from_dlpack(a)
    np.testing.assert_allclose(a.numpy(), b.numpy())

    a[1:] = 1.0
    np.testing.assert_allclose(a.numpy(), b.numpy())

    c = torch.randn([2, 3]).to(torch_device)
    d = hidet.from_dlpack(c)
    np.testing.assert_allclose(c.cpu().numpy(), d.cpu().numpy())

    c[1:] = 1.0
    np.testing.assert_allclose(c.cpu().numpy(), d.cpu().numpy())

    # round-trip for bool torch tensor
    a = torch.empty(10, dtype=torch.bool, device=torch_device)
    b = hidet.from_dlpack(a)
    assert b.device.kind == device and b.dtype.name == 'bool'
    c = b.torch()
    assert a.device == c.device and a.dtype == c.dtype

    # round-trip for bool numpy tensor
    a = np.empty(shape=[10], dtype=np.bool)
    b = hidet.from_dlpack(a)
    assert b.device.kind == 'cpu' and b.dtype.name == 'bool'
    c = b.numpy()
    assert a.device == c.device and a.dtype == c.dtype


def test_to_dlpack(device):
    torch_device = device_to_torch(device)

    a = hidet.randn([2, 3]).cpu()
    b = torch.from_dlpack(a)
    np.testing.assert_allclose(a.numpy(), b.numpy())

    b[1:] = 1.0
    np.testing.assert_allclose(a.numpy(), b.numpy())

    d = hidet.randn([2, 3]).to(device=device)
    e = torch.from_dlpack(d)
    np.testing.assert_allclose(d.cpu().numpy(), e.cpu().numpy())

    e[1:] = 1.0
    np.testing.assert_allclose(d.cpu().numpy(), e.cpu().numpy())


def test_type_conversion():
    a = hidet.randn([2, 3], dtype=hidet.float32)
    assert (a + 1).dtype == hidet.float32
    assert (a + 1.0).dtype == hidet.float32
    assert (a + hidet.float64(1.0)).dtype == hidet.float32

    a = hidet.randn([2, 3], dtype=hidet.int32)
    assert (a + 1).dtype == hidet.int32
    assert (a + 1.0).dtype == hidet.float32
    assert (a + hidet.float64(1.0)).dtype == hidet.float64
