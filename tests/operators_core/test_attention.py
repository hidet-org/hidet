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
import torch
import pytest
import numpy
import hidet
from hidet.graph.ops.attention import attention
from hidet import ops


@pytest.mark.requires_cuda
@pytest.mark.parametrize("shape", [[2, 512, 512, 8, 128], [2, 435, 179, 8, 64]])
@pytest.mark.parametrize("dtype", ['float16', 'bfloat16'])
def test_attn_mask_add(shape, dtype):
    bs, s_q, s_kv, h, d = shape

    def attention_layer():
        q = hidet.symbol([bs, h, s_q, d], dtype=dtype, device='cuda')
        k = hidet.symbol([bs, h, d, s_kv], dtype=dtype, device='cuda')
        v = hidet.symbol([bs, h, s_kv, d], dtype=dtype, device='cuda')
        mask = hidet.symbol([bs, h, s_q, s_kv], dtype=dtype, device='cuda')
        qk = ops.matmul(q, k)
        qk_masked = qk + mask
        sm = ops.softmax(qk_masked, axis=-1)
        out = ops.matmul(sm, v)
        return hidet.trace_from(out, [q, k, v, mask])

    graph = attention_layer()
    q = hidet.randn([bs, h, s_q, d], dtype=dtype, device='cuda') / 10
    k = hidet.randn([bs, h, d, s_kv], dtype=dtype, device='cuda') / 10
    v = hidet.from_torch(torch.rand([bs, h, s_kv, d], dtype=getattr(torch, dtype), device='cuda') - 0.5)
    mask = hidet.randn([bs, h, s_q, s_kv], dtype=dtype, device='cuda') / 10

    cc1 = attention(q, k, v, mask)
    cc2 = graph(q, k, v, mask)

    if dtype == 'bfloat16':
        cc1 = cc1.to(dtype='float32')
        cc2 = cc2.to(dtype='float32')

    tol = 1e-2

    numpy.testing.assert_allclose(cc1.cpu().numpy(), cc2.cpu().numpy(), atol=tol, rtol=tol)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("shape", [[2, 1024, 1024, 8, 128], [2, 667, 775, 8, 64]])
@pytest.mark.parametrize("dtype", ['float16', 'bfloat16'])
def test_attn(shape, dtype):
    bs, s_q, s_kv, h, d = shape

    def attention_layer():
        q = hidet.symbol([bs, h, s_q, d], dtype=dtype, device='cuda')
        k = hidet.symbol([bs, h, d, s_kv], dtype=dtype, device='cuda')
        v = hidet.symbol([bs, h, s_kv, d], dtype=dtype, device='cuda')
        qk = ops.matmul(q, k)
        sm = ops.softmax(qk, axis=-1)
        out = ops.matmul(sm, v)
        return hidet.trace_from(out, [q, k, v])

    graph = attention_layer()
    # need to `optimize` to make sure the matmul is resolved to `matmul_f16_cute`,
    # which is necessary for the bfloat16 case
    graph = hidet.graph.optimize(graph)
    q = hidet.randn([bs, h, s_q, d], dtype=dtype, device='cuda')
    k = hidet.randn([bs, h, d, s_kv], dtype=dtype, device='cuda')
    v = hidet.randn([bs, h, s_kv, d], dtype=dtype, device='cuda')

    cc1 = attention(q, k, v)
    cc2 = graph(q, k, v)

    if dtype == 'bfloat16':
        cc1 = cc1.to(dtype='float32')
        cc2 = cc2.to(dtype='float32')

    # If we set the tolerance to 0.5, for the bfloat16 case, the test is kind of flaky:
    # E       Mismatched elements: 13 / 683008 (0.0019%)
    # E       Max absolute difference: 0.8417969
    # E       Max relative difference: 1.3478313e+11
    tol = 0.5 if dtype == 'float16' else 1.0

    numpy.testing.assert_allclose(cc1.cpu().numpy(), cc2.cpu().numpy(), atol=tol, rtol=tol)


if __name__ == '__main__':
    pytest.main([__file__])
