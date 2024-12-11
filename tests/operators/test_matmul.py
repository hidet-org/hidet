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
import torch

import hidet
from hidet import ops
from hidet.testing import check_binary, check_binary_dynamic, check_torch_binary, check_torch_binary_with_inputs


# @pytest.mark.skip(reason="when running matmul_x86 multiple times, it will produce wrong result. need fix.")
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
    if hidet.option.cuda.get_arch_pair() < (8, 0) and mma in ['wmma', 'mma'] and dtype == 'float32':
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
def test_batch_matmul_dynamic(a_shape, b_shape, dtype: str, mma: str):
    if hidet.option.cuda.get_arch_pair() < (8, 0) and mma in ['wmma', 'mma'] and dtype == 'float32':
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


# This test checks the correctness of the implementation of using f16/f32 accumulator of tensor's core mma
def test_matmul_fp16_fp32():
    from hidet.graph.ops.matmul.matmul_f16 import matmul_f16

    m, n, k = 128, 128, 65536 + 8
    a = torch.ones((m, k), dtype=torch.float16, device="cuda")
    a = torch.concat([a, -a], dim=1)
    b = torch.ones((k * 2, n), dtype=torch.float16, device="cuda")

    check_torch_binary_with_inputs(
        a,
        b,
        lambda x, y: torch.matmul(x, y),
        lambda x, y: ops.squeeze(matmul_f16(x, y, acc_dtype="float32"), 0),
        atol=1e-2,
        rtol=1e-2,
    )

    with pytest.raises(AssertionError) as e_info:
        e_info
        check_torch_binary_with_inputs(
            a,
            b,
            lambda x, y: torch.matmul(x, y),
            lambda x, y: ops.squeeze(matmul_f16(x, y, acc_dtype="float16"), 0),
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 4096, 4096], [4096, 4096]],
        [[1, 4096, 4096], [4096, 4096 + 2]],
        [[1, 120000, 320], [320, 768]],
        [[1, 120000, 768], [768, 320]],
        [[1, 192, 256], [256, 128]],
        [[1, 128, 128 + 4], [128 + 4, 128]],
        [[1, 128, 128 + 2], [128 + 2, 128]],
        [[1, 128, 128 + 2], [128 + 2, 128 - 2]],
        [[1, 128, 128], [128, 128 - 4]],
    ],
)
def test_matmul_fp16_cute(a_shape, b_shape):
    from hidet.graph.ops.matmul.matmul_f16_cute_experimental import matmul_f16_cute as matmul_f16

    # hidet.option.cache_dir("./debug_matmul")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    check_torch_binary(
        a_shape,
        b_shape,
        torch_func=lambda x, y: torch.matmul(x, y),
        hidet_func=lambda x, y: ops.squeeze(matmul_f16(x, y), 0),
        device='cuda',
        dtype='float16',
        atol=1e-2,
        rtol=1e-2,
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


@pytest.mark.parametrize("a_shape, b_shape", [[[1, 128, 128], [128, 128]]])
@pytest.mark.parametrize("dtype, tol", [("float32", 1e-5), ("float16", 1e-1), ("bfloat16", 5e-1)])
def test_matmul_cublas(a_shape, b_shape, dtype, tol):
    check_torch_binary(
        a_shape,
        b_shape,
        torch_func=lambda x, y: torch.matmul(x, y),
        hidet_func=lambda x, y: ops.matmul_cublas(x, y),
        device="cuda",
        dtype=dtype,
        atol=tol,
        rtol=tol,
    )


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 128, 128], [128, 128]],
        [[1, 128, 128 + 4], [128, 128 + 4]],
        [[1, 128, 128 + 2], [128, 128 + 2]],
        [[1, 128, 128 + 2], [128 - 2, 128 + 2]],
        [[1, 128, 128], [128 - 4, 128]],
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('parallel_k', ['disabled', 'default', 2, 3])
def test_matmul_nt(a_shape, b_shape, dtype, parallel_k):
    from hidet.testing import assert_torch_allclose

    with hidet.option.context():
        hidet.option.parallel_k(parallel_k)
        a = torch.randn(*a_shape, dtype=dtype, device='cuda')
        b = torch.randn(*b_shape, dtype=dtype, device='cuda')
        c_correct = torch.matmul(a, torch.transpose(b, 0, 1))
        ahi = hidet.from_torch(a)
        bhi = hidet.from_torch(b)
        ahi_symbol = hidet.symbol_like(ahi)
        bhi_symbol = hidet.symbol_like(bhi)
        cc = ops.matmul_nt(ahi_symbol, bhi_symbol)
        graph = hidet.graph.trace_from(cc, inputs=[ahi_symbol, bhi_symbol])
        graph_opt = hidet.graph.optimize(graph)
        c_hi = graph_opt(ahi, bhi)
        assert_torch_allclose(c_hi.cpu(), c_correct.cpu(), atol=1e-1, rtol=1e-1)


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
def test_matmul_bf16(a_shape, b_shape):
    a = torch.randn(*a_shape, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(*b_shape, dtype=torch.bfloat16, device='cuda')
    c_correct = torch.matmul(a, b).to(dtype=torch.float32)
    ahi = hidet.from_torch(a)
    bhi = hidet.from_torch(b)
    ahi_symbol = hidet.symbol_like(ahi)
    bhi_symbol = hidet.symbol_like(bhi)
    cc = ops.matmul(ahi_symbol, bhi_symbol)
    graph = hidet.graph.trace_from(cc, inputs=[ahi_symbol, bhi_symbol])
    graph_opt = hidet.graph.optimize(graph)
    c_hi = graph_opt(ahi, bhi).to(dtype='float32')
    np.testing.assert_allclose(c_hi.cpu().numpy(), c_correct.cpu().numpy(), atol=1e-1, rtol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__])
