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

from hidet import ops
from hidet.testing import check_unary, check_binary


@pytest.mark.parametrize("a_shape, b_shape, dtype", [[[33, 44], [44], "bool"]])
def test_and(a_shape, b_shape, dtype):
    # without broadcast
    check_binary(a_shape, a_shape, lambda a, b: np.logical_and(a, b), lambda a, b: ops.logical_and(a, b), dtype=dtype)
    # with broadcast
    check_binary(a_shape, b_shape, lambda a, b: np.logical_and(a, b), lambda a, b: ops.logical_and(a, b), dtype=dtype)


if __name__ == '__main__':
    pytest.main([__file__])
