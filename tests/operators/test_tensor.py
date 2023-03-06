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
import numpy as np
import torch
import hidet


def test_from_dlpack():
    a = torch.randn([2, 3])
    b = hidet.from_dlpack(a)
    np.testing.assert_allclose(a.numpy(), b.numpy())

    a[1:] = 1.0
    np.testing.assert_allclose(a.numpy(), b.numpy())

    c = torch.randn([2, 3]).cuda()
    d = hidet.from_dlpack(c)
    np.testing.assert_allclose(c.cpu().numpy(), d.cpu().numpy())

    c[1:] = 1.0
    np.testing.assert_allclose(c.cpu().numpy(), d.cpu().numpy())


def test_to_dlpack():
    a = hidet.randn([2, 3]).cpu()
    b = torch.from_dlpack(a)
    np.testing.assert_allclose(a.numpy(), b.numpy())

    b[1:] = 1.0
    np.testing.assert_allclose(a.numpy(), b.numpy())

    d = hidet.randn([2, 3]).cuda()
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
