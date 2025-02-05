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


@pytest.mark.parametrize('n', [3, 7])
@pytest.mark.parametrize('m', [3, 7])
@pytest.mark.parametrize('k', [-1, 1])
def test_tri(n, m, k):
    import numpy as np

    a = hidet.ops.tri(n, m, k)
    b = np.tri(n, m, k)
    assert np.allclose(a.numpy(), b)
