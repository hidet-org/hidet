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
from hidet.graph import ops
from hidet.testing import check_unary


@pytest.mark.parametrize("op", ["triu", "tril"])
@pytest.mark.parametrize("shape, diagonal", [[(3, 4), -1], [(4, 3), 2], [(3, 1, 4, 5), 0], [(3, 3), -99]])
def test_tri(op, shape, diagonal, device):
    numpy_op, hidet_op = getattr(np, op), getattr(ops, op)
    check_unary(
        shape,
        numpy_op=lambda x: numpy_op(x, diagonal),
        hidet_op=lambda x: hidet_op(x, diagonal),
        dtype="float32",
        atol=1e-6,
        rtol=1e-6,
        device=device,
    )
