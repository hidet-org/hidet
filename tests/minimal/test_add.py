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
import hidet
from hidet.testing import device_to_torch


def test_add(device):
    torch_device = device_to_torch(device)
    a = hidet.randn([10], device=torch_device)
    b = hidet.randn([10], device=torch_device)
    c = a + b
    c_np = a.cpu().numpy() + b.cpu().numpy()
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c_np, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
