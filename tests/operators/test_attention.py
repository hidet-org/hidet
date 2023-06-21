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
from hidet.graph.ops.attention import attention
from hidet import ops


@pytest.mark.parametrize("shape", [[2, 512, 512, 8, 128], [2, 435, 179, 8, 64]])
def test_attn_mask_add(shape):
    bs, s_q, s_kv, h, d = shape

    def attention_layer():
        q = hidet.symbol([bs, h, s_q, d], dtype='float16', device='cuda')
        k = hidet.symbol([bs, h, d, s_kv], dtype='float16', device='cuda')
        v = hidet.symbol([bs, h, s_kv, d], dtype='float16', device='cuda')
        mask = hidet.symbol([bs, h, s_q, s_kv], dtype='float16', device='cuda')
        qk = ops.matmul(q, k)
        qk_masked = qk + mask
        sm = ops.softmax(qk_masked, axis=-1)
        out = ops.matmul(sm, v)
        return hidet.trace_from(out, [q, k, v, mask])

    graph = attention_layer()
    q = hidet.randn([bs, h, s_q, d], dtype='float16', device='cuda')
    k = hidet.randn([bs, h, d, s_kv], dtype='float16', device='cuda')
    v = hidet.randn([bs, h, s_kv, d], dtype='float16', device='cuda')
    mask = hidet.randn([bs, h, s_q, s_kv], dtype='float16', device='cuda')

    cc1 = attention(q, k, v, mask)
    cc2 = graph(q, k, v, mask)

    numpy.testing.assert_allclose(cc1.cpu().numpy(), cc2.cpu().numpy(), atol=0.5, rtol=0.5)


@pytest.mark.parametrize("shape", [[2, 1024, 1024, 8, 128], [2, 667, 775, 8, 64]])
def test_attn(shape):
    bs, s_q, s_kv, h, d = shape

    def attention_layer():
        q = hidet.symbol([bs, h, s_q, d], dtype='float16', device='cuda')
        k = hidet.symbol([bs, h, d, s_kv], dtype='float16', device='cuda')
        v = hidet.symbol([bs, h, s_kv, d], dtype='float16', device='cuda')
        qk = ops.matmul(q, k)
        sm = ops.softmax(qk, axis=-1)
        out = ops.matmul(sm, v)
        return hidet.trace_from(out, [q, k, v])

    graph = attention_layer()
    q = hidet.randn([bs, h, s_q, d], dtype='float16', device='cuda')
    k = hidet.randn([bs, h, d, s_kv], dtype='float16', device='cuda')
    v = hidet.randn([bs, h, s_kv, d], dtype='float16', device='cuda')

    cc1 = attention(q, k, v)
    cc2 = graph(q, k, v)

    numpy.testing.assert_allclose(cc1.cpu().numpy(), cc2.cpu().numpy(), atol=0.5, rtol=0.5)


if __name__ == '__main__':
    pytest.main([__file__])
