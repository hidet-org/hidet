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

import hidet
from hidet import ops
from hidet.testing import check_binary


@pytest.mark.parametrize(
    "a_shape, b_shape, dtype", [[[1, 333, 444], [1, 444, 555], "float32"], [[1, 333, 444], [1, 444, 555], "float16"]]
)
@pytest.mark.parametrize('mma', ['simt', 'mma'])
def test_batch_matmul(a_shape, b_shape, dtype, mma):
    if hidet.cuda.compute_capability() < (8, 0) and mma in ['wmma', 'mma'] and dtype == 'float32':
        pytest.skip('wmma and mma for float32 will triger hidet to use tf32, which is only supported on sm80 and above')
    tolerance = {('float16', 'simt'): 0.5, ('float16', 'mma'): 0.5, ('float32', 'simt'): 1e-4, ('float32', 'mma'): 0.05}
    tol = tolerance[(dtype, mma)]
    check_binary(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.batch_matmul(x, y, mma=mma),
        device='cuda',
        dtype=dtype,
        atol=tol,
        rtol=tol,
    )


@pytest.mark.parametrize(
    "a_shape, b_shape, dtype",
    [[[1, 128, 128], [128, 128], "float32"], [[333, 444], [444], "float32"], [[129, 443], [443], "complex64"]],
)
def test_matmul(a_shape, b_shape, dtype):
    check_binary(
        a_shape, b_shape, lambda x, y: np.matmul(x, y), lambda x, y: ops.matmul(x, y), dtype=dtype, atol=1e-4, rtol=1e-4
    )


if __name__ == '__main__':
    pytest.main([__file__])
