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


def test_add():
    a = hidet.randn([10], device='cuda')
    b = hidet.randn([10], device='cuda')
    c = a + b
    c_np = a.cpu().numpy() + b.cpu().numpy()
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c_np, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
