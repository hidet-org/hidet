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
from hidet import ops
import torch

from hidet.testing import check_torch_unary
from hidet.graph.frontend.torch.register_modules import HidetSigmoid


@pytest.mark.parametrize("shape", [[10, 10, 10], [15, 15, 15]])
def test_sigmoid(shape):
    sigmoid = HidetSigmoid(torch.nn.Sigmoid())
    check_torch_unary(shape, lambda x: torch.sigmoid(x), lambda x: sigmoid(x), dtype='float32', atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
