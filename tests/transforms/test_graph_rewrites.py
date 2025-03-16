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

import hidet
import torch
import numpy as np
import pytest
from hidet.graph.transforms.graph_patterns import matmul_patterns
from hidet import ops
from hidet.graph.ops.matmul.matmul_f16_cute import MatmulF16CuteOp
from hidet.graph.ops.matmul.matmul_f16_cute_experimental import MatmulF16CuteOp as MatmulF116CuteopV2


@pytest.mark.requires_cuda
@pytest.mark.parametrize("a_shape, b_shape", [[[512, 768], [1024, 768]]])
def test_matmul_transpose_rewrite(a_shape, b_shape):
    # Create PyTorch tensors
    a = torch.randn(*a_shape, dtype=torch.float16, device='cuda')
    b = torch.randn(*b_shape, dtype=torch.float16, device='cuda')

    # Compute reference output
    c_correct = torch.matmul(a, b.transpose(0, 1)).to(dtype=torch.float32)

    # Convert to Hidet tensors
    ahi = hidet.from_torch(a)
    bhi = hidet.from_torch(b)

    # Create symbolic tensors for both a and b
    ahi_symbol = hidet.symbol_like(ahi)
    bhi_symbol = hidet.symbol_like(bhi)

    # Build the computation graph with transpose and matmul operations
    b_t = ops.transpose(bhi_symbol, (1, 0))
    c_hi = ops.matmul(ahi_symbol, b_t)

    # Trace and optimize the graph
    graph = hidet.graph.trace_from(c_hi, inputs=[ahi_symbol, bhi_symbol])
    graph_opt = hidet.graph.optimize(graph)

    # Assert that graph optimization converted matmul+transpose to matmul_nt
    assert isinstance(
        graph_opt.outputs[0].trace[0], (MatmulF16CuteOp, MatmulF116CuteopV2)
    ), "Expected matmul_f16_cute at the beginning of output's trace"
    assert graph_opt.outputs[0].trace[0].attrs['transpose_b'] is True, "Expected transpose_b=True in matmul"

    # Run the optimized graph
    c_hi_opt = graph_opt(ahi, bhi)

    # Compare with reference output
    np.testing.assert_allclose(c_hi_opt.cpu().numpy(), c_correct.cpu().numpy(), atol=1e-1, rtol=1e-1)
