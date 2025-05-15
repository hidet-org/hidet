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


@pytest.mark.requires_cuda
@pytest.mark.parametrize(
    "a_shape, b_shape, use_transpose",
    [
        # Basic test case - similar to QWen example (standard matmul)
        [[1296, 1, 1024], [3840, 1024], True],
        # Test with different batch dimensions
        [[3, 32, 128], [128, 256], False],
        # Test with matmul_nt - B needs to be in [n, k] format
        [[3, 10, 10, 512], [128, 512], True],
    ],
)
def test_batched_matmul_flatten_rule(a_shape, b_shape, use_transpose):
    # Create PyTorch tensors with controlled values and reproducible randomness
    torch.manual_seed(0)  # For reproducibility
    low, high = -3, 3
    scale = 5.0  # Use float division to get fractional values
    a = torch.randint(low=low, high=high, size=a_shape, device='cuda').to(torch.float16) / scale
    b = torch.randint(low=low, high=high, size=b_shape, device='cuda').to(torch.float16)

    # 1. Calculate reference using PyTorch
    if use_transpose:
        # For matmul_nt, B has shape [n, k], we need to transpose to [k, n] for torch.matmul
        c_correct = torch.matmul(a, torch.transpose(b, 0, 1))
    else:
        # For standard matmul, B already has shape [k, n]
        c_correct = torch.matmul(a, b)

    # 2. Set up Hidet computation
    ahi = hidet.from_torch(a)
    bhi = hidet.from_torch(b)
    ahi_symbol = hidet.symbol_like(ahi)
    bhi_symbol = hidet.symbol_like(bhi)

    if use_transpose:
        # Verify tensor shapes match expected dimensions
        assert b_shape[1] == a_shape[-1]
        c_hi = ops.matmul_nt(ahi_symbol, bhi_symbol)  # B is transposed
    else:
        # Verify tensor shapes match expected dimensions
        assert b_shape[0] == a_shape[-1]
        c_hi = ops.matmul(ahi_symbol, bhi_symbol)  # Standard matmul

    # 3. Trace and optimize
    graph = hidet.graph.trace_from(c_hi, inputs=[ahi_symbol, bhi_symbol])
    graph_opt = hidet.graph.optimize(graph)

    # 4. Run optimized graph
    c_hi_opt = graph_opt(ahi, bhi)

    # Check if reshape operations appear in the optimized graph
    reshape_ops_found = False
    for op in graph_opt.nodes:
        if op.__class__.__name__ == 'ReshapeOp':
            reshape_ops_found = True
            break

    assert reshape_ops_found, "Optimization was not applied - no reshape operations found in the optimized graph"

    # 5. Verify results match
    c_hi_opt_np = c_hi_opt.cpu().numpy()
    c_correct_np = c_correct.cpu().numpy()

    # Test shapes match exactly
    assert c_hi_opt.shape == c_correct.shape, f"Shape mismatch: {c_hi_opt.shape} vs {c_correct.shape}"

    # Test values match within tolerance
    np.testing.assert_allclose(c_hi_opt_np, c_correct_np, atol=1e-1, rtol=1e-1)
