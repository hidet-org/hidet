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
from hidet import ops


@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dtype", ['int8', 'int16'])
@pytest.mark.parametrize("dims", [[-1], [0]])
def test_symmetric_quant(w, dtype, dims):
    w = hidet.randn((w, w), dtype='float32')
    wq, scale = ops.symmetric_quantize(w, dtype, dims)
    w1 = ops.symmetric_dequantize(wq, scale, dims)
    assert np.allclose(w.numpy(), w1.numpy(), atol=1e-1, rtol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__])
