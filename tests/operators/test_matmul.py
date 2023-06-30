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
from hidet.testing import check_binary, check_binary_dynamic


@pytest.mark.skip(reason="when running matmul_x86 multiple times, it will produce wrong result. need fix.")
@pytest.mark.parametrize("a_shape, b_shape", [[[333, 444], [444, 555]], [[133, 1], [1, 177]]])
def test_matmul_x86(a_shape, b_shape):
    # TODO: Doesn't support broadcasting yet; need to add it later?
    check_binary(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.matmul_x86(x, y) - ops.matmul_x86(x, y) + ops.matmul_x86(x, y),
        dtype="float32",
        atol=1e-4,
        rtol=1e-4,
        device="cpu",
    )


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
    [
        [[1, ("n", 333), ("m", 444)], [1, ("m", 444), ("k", 555)], "float32"],
        [[("b", 1), ("m", 333), ("k", 444)], [("b", 1), ("k", 444), ("n", 555)], "float16"],
    ],
)
@pytest.mark.parametrize('mma', ['simt', 'mma'])
def test_batch_matmul_dynamic(a_shape, b_shape, dtype, mma):
    if hidet.cuda.compute_capability() < (8, 0) and mma in ['wmma', 'mma'] and dtype == 'float32':
        pytest.skip('wmma and mma for float32 will triger hidet to use tf32, which is only supported on sm80 and above')
    tolerance = {('float16', 'simt'): 0.5, ('float16', 'mma'): 0.5, ('float32', 'simt'): 1e-4, ('float32', 'mma'): 0.05}
    tol = tolerance[(dtype, mma)]
    check_binary_dynamic(
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


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 128, 128], [128, 128]],
        [[1, 128, 128 + 4], [128 + 4, 128]],
        [[1, 128, 128 + 2], [128 + 2, 128]],
        [[1, 128, 128 + 2], [128 + 2, 128 - 2]],
        [[1, 128, 128], [128, 128 - 4]],
    ],
)
def test_matmul_fp16(a_shape, b_shape):
    from hidet.graph.ops.matmul.matmul_f16 import matmul_f16

    check_binary(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.squeeze(matmul_f16(x, y), 0),
        dtype='float16',
        atol=1e-1,
        rtol=1e-1,
        device='cuda',
    )


@pytest.mark.parametrize("a_shape, b_shape", [[[1, 128, ("s", 128)], [("s", 128), 128]]])
def test_matmul_fp16_dynamic(a_shape, b_shape):
    from hidet.graph.ops.matmul.matmul_f16 import matmul_f16

    check_binary_dynamic(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.squeeze(matmul_f16(x, y), 0),
        dtype='float16',
        atol=1e-1,
        rtol=1e-1,
        device='cuda',
    )


if __name__ == '__main__':
    pytest.main([__file__])
