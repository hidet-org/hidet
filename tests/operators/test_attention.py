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
import math
import pytest
import numpy
import hidet
from hidet.graph.ops.definitions.attention import attention
from hidet import ops


def test_attn_mask_add():
    def attention_layer():
        q = hidet.symbol([2, 1, 512, 64], dtype='float16', device='cuda')
        k = hidet.symbol([2, 1, 64, 512], dtype='float16', device='cuda')
        v = hidet.symbol([2, 1, 512, 64], dtype='float16', device='cuda')
        mask = hidet.symbol([2, 1, 512, 512], dtype='float16', device='cuda')
        qk = ops.matmul(q, k)
        qk_masked = qk + mask
        sm = ops.softmax(qk_masked, axis=-1)
        out = ops.matmul(sm, v)
        hidet.graph.PassContext().set_use_attention(False)
        return hidet.graph.optimize(hidet.trace_from(out, [q, k, v, mask]))

    graph = attention_layer()
    q = hidet.randn([2, 1, 512, 64], dtype='float16', device='cuda')
    k = hidet.randn([2, 1, 64, 512], dtype='float16', device='cuda')
    v = hidet.randn([2, 1, 512, 64], dtype='float16', device='cuda')
    mask = hidet.randn([2, 1, 512, 512], dtype='float16', device='cuda')

    cc1 = attention(q, k, v, mask)
    cc2 = graph(q, k, v, mask)

    numpy.testing.assert_allclose(cc1.cpu().numpy(), cc2.cpu().numpy(), atol=1e-2, rtol=1e-2)


def test_attn():
    def attention_layer():
        q = hidet.symbol([3, 1, 2, 1024, 128], dtype='float16', device='cuda')
        k = hidet.symbol([3, 1, 2, 128, 1024], dtype='float16', device='cuda')
        v = hidet.symbol([3, 1, 2, 1024, 128], dtype='float16', device='cuda')
        qk = ops.matmul(q, k)
        sm = ops.softmax(qk, axis=-1)
        out = ops.matmul(sm, v)
        hidet.graph.PassContext().set_use_attention(False)
        return hidet.graph.optimize(hidet.trace_from(out, [q, k, v]))

    graph = attention_layer()
    q = hidet.randn([3, 1, 2, 1024, 128], dtype='float16', device='cuda')
    k = hidet.randn([3, 1, 2, 128, 1024], dtype='float16', device='cuda')
    v = hidet.randn([3, 1, 2, 1024, 128], dtype='float16', device='cuda')

    cc1 = attention(q, k, v)
    cc2 = graph(q, k, v)

    numpy.testing.assert_allclose(cc1.cpu().numpy(), cc2.cpu().numpy(), atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
