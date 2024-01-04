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


def test_as_torch_tensor():
    """
    test __torch_func__ protocol
    """
    a = hidet.randn([32, 32], dtype='float16', device='cuda')
    b = torch.abs(a)
    c = hidet.ops.abs(a)
    torch.testing.assert_close(b, c.torch())


if __name__ == '__main__':
    pytest.main([__file__])
