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

from hidet.graph.ops import quant


@pytest.mark.parametrize('shape', [[32, 32], [64, 64]])
@pytest.mark.parametrize('dim', [0, 1])
@pytest.mark.parametrize('dtype', ['int8', 'int16'])
@pytest.mark.parametrize('device', ['cuda'])
def test_symmetric_quant(shape, dim, dtype, device):
    a = hidet.randn(shape, dtype='float16', device=device)
    aq, scale = quant.symmetric_quantize(a, dims=dim, quant_type=dtype)
    a1 = quant.symmetric_dequantize(aq, scale, dims=dim)
    assert np.allclose(a.cpu().numpy(), a1.cpu().numpy(), atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(['x_shape', 'w_shape'], [[[2, 32, 32], [32, 32]], [[1, 64, 64], [64, 128]]])
def test_symmetric_quant_matmul(x_shape, w_shape):
    x = hidet.randn(x_shape, dtype='float16', device='cuda')
    w = hidet.randn(w_shape, dtype='float16', device='cuda')
    wq, scale = quant.symmetric_quantize(w, dims=0, quant_type='int8')
    yq = quant.symmetric_quant_matmul(x, wq, scale)
    y = ops.matmul(x, w)
    assert np.allclose(y.cpu().numpy(), yq.cpu().numpy(), atol=2e-1, rtol=2e-1)


if __name__ == "__main__":
    pytest.main([__file__])
