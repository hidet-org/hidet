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
import math
import hidet
from hidet import ops
from hidet.cuda.cudnn import cudnnDataType


@pytest.mark.parametrize(
    "n, c, h, w, k, p, q, r, s, padding, stride, dilations",
    [
        [1, 3, 32, 32, 12, 30, 30, 3, 3, [0, 0], [1, 1], [1, 1]],  # kernel 3,
        [2, 3, 32, 32, 12, 11, 6, 7, 7, [1, 2], [2, 3], [2, 3]],  # kernel 7, batch size 2
        [1, 3, 32, 32, 12, 16, 11, 1, 1, [0, 0], [2, 3], [1, 1]],  # kernel 1,
    ],
)
@pytest.mark.parametrize(
    'dtype, compute_type, tol',
    [(hidet.float32, cudnnDataType.CUDNN_DATA_FLOAT, 1e-5), (hidet.float64, cudnnDataType.CUDNN_DATA_DOUBLE, 1e-8)],
)
def test_cudnn_conv2d(n, c, h, w, k, p, q, r, s, dtype, compute_type, padding, stride, dilations, tol):
    tx = tw = ty = dtype
    pad_dim1, pad_dim2 = padding
    str_dim1, str_dim2 = stride
    dil_dim1, dil_dim2 = dilations

    tensor_x = hidet.randn((n, c, h, w), device='cuda', dtype=tx)
    tensor_w = hidet.randn((k, c, r, s), device='cuda', dtype=tw)
    tensor_y = hidet.empty((n, k, p, q), device='cuda', dtype=ty)

    golden = ops.conv2d(
        tensor_x, tensor_w, stride=(str_dim1, str_dim2), dilations=(dil_dim1, dil_dim2), padding=(pad_dim1, pad_dim2)
    )
    hidet.cuda.cudnn.conv2d(
        n,
        c,
        h,
        w,
        k,
        r,
        s,
        p,
        q,
        tensor_x,
        tensor_w,
        tensor_y,
        tx,
        tw,
        ty,
        compute_type,
        pad_dim1,
        pad_dim2,
        str_dim1,
        str_dim2,
        dil_dim1,
        dil_dim2,
    )

    hidet.utils.assert_close(actual=tensor_y, expected=golden, rtol=tol, atol=tol)


if __name__ == '__main__':
    pytest.main([__file__])
