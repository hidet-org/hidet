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
import numpy as np
import pytest
import torch

import hidet
from hidet import ops
from hidet.testing import check_binary, check_binary_dynamic, check_torch_binary, check_torch_binary_with_inputs


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


@pytest.mark.requires_cuda
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
        lambda x, y: ops.cuda_batch_matmul(x, y, mma=mma),
        device='cuda',
        dtype=dtype,
        atol=tol,
        rtol=tol,
    )


@pytest.mark.requires_cuda
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
        lambda x, y: ops.cuda_batch_matmul(x, y, mma=mma),
        device='cuda',
        dtype=dtype,
        atol=tol,
        rtol=tol,
    )


@pytest.mark.requires_cuda
@pytest.mark.parametrize(
    "a_shape, b_shape, dtype",
    [[[1, 128, 128], [128, 128], "float32"], [[333, 444], [444], "float32"], [[129, 443], [443], "complex64"]],
)
def test_matmul(a_shape, b_shape, dtype, device):
    check_binary(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.matmul(x, y),
        dtype=dtype,
        atol=1e-4,
        rtol=1e-4,
        device=device,
    )


@pytest.mark.requires_cuda
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


@pytest.mark.requires_cuda_hopper
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
def test_matmul_fp16_sm90(a_shape, b_shape):
    from hidet.graph.ops.matmul.matmul_f16_sm90 import matmul_f16_sm90

    check_binary(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.squeeze(matmul_f16_sm90(x, y, is_a_shared=True), 0),
        dtype='float16',
        atol=1e-1,
        rtol=1e-1,
        device='cuda',
    )

    check_binary(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.squeeze(matmul_f16_sm90(x, y, is_a_shared=False), 0),
        dtype='float16',
        atol=1e-1,
        rtol=1e-1,
        device='cuda',
    )


# This test checks the correctness of the implementation of using f16/f32 accumulator of tensor's core mma
@pytest.mark.requires_cuda
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


@pytest.mark.requires_cuda
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
    k = b_shape[0]

    check_torch_binary(
        a_shape,
        b_shape,
        torch_func=lambda x, y: torch.matmul(x / k, y),
        hidet_func=lambda x, y: matmul_f16(x / k, y),
        device='cuda',
        dtype='float16',
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.requires_cuda
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


@pytest.mark.requires_cuda
@pytest.mark.parametrize(
    "a_shape, b_shape", [[[256, 128], [256, 128]], [[4096, 4096], [4096, 4096]], [[512, 256], [128, 256]]]
)
def test_matmul_f8(a_shape, b_shape):
    import hidet
    from hidet.testing.torch_utils import device_to_torch
    from hidet.ir.dtypes import f16, f8e4m3
    from hidet.graph.frontend.torch.utils import dtype_to_torch
    from hidet.graph.ops.matmul.matmul_f8 import matmul_f8

    hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    device, rtol, atol = "cuda", 1e-2, 1e-2
    gen_dtype = dtype_to_torch(f16)
    dtype = dtype_to_torch(f8e4m3)

    torch_device = device_to_torch(device)
    torch_a = torch.randint(-3, 3, a_shape, dtype=gen_dtype, device=torch_device).to(dtype=dtype)
    torch_b = torch.randint(-3, 3, b_shape, dtype=gen_dtype, device=torch_device).to(dtype=dtype)
    hidet_a = hidet.from_torch(torch_a)
    hidet_b = hidet.from_torch(torch_b)
    torch_result: torch.Tensor = torch._scaled_mm(
        torch_a,
        torch_b.T,
        out_dtype=dtype,
        scale_a=torch.tensor(1.0, device=device),
        scale_b=torch.tensor(1.0, device=device),
    )
    hidet_result: hidet.Tensor = matmul_f8(hidet_a, hidet_b)

    torch.testing.assert_close(
        actual=hidet_result.torch().to(dtype=torch.float16),
        expected=torch_result.to(dtype=torch.float16),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.requires_cuda
@pytest.mark.parametrize(
    "a_shape, b_shape", [[[4096, 4096], [4096, 4096]], [[2048, 256], [4096, 256]], [[16384, 7168], [1536, 7168]]]
)
@pytest.mark.parametrize("group_m, group_n, group_k", [[1, 128, 128]])
def test_matmul_f8_scaled(a_shape, b_shape, group_m, group_n, group_k):
    # input a_shape is (M,K) and b_shape is (N,K)
    # vLLM uses this exact scaled matmul configuration
    # m16384n1536k7168 is used in DSR1
    import hidet
    from hidet.testing.torch_utils import device_to_torch
    from hidet.ir.dtypes import f16, f8e4m3, f32, bf16
    from hidet.graph.frontend.torch.utils import dtype_to_torch
    from hidet.graph.ops.matmul.matmul_f8 import matmul_f8_scaled

    hidet.option.search_space(2)

    # dtype settings
    device, rtol, atol, gen_dtype, dtype = (
        device_to_torch("cuda"),
        1e-2,
        1e-2,
        dtype_to_torch(f32),
        dtype_to_torch(f8e4m3),
    )
    output_dtype = bf16

    M, K = a_shape
    N = b_shape[0]
    num_group_M, num_group_N, num_group_K = M // group_m, N // group_n, K // group_k

    # generate random input
    torch_a = torch.randint(-5, 5, a_shape, dtype=gen_dtype, device=device)
    torch_b = torch.randint(-5, 5, b_shape, dtype=gen_dtype, device=device)
    torch_scale_a = torch.randint(1, 5, [num_group_K, num_group_M], dtype=dtype_to_torch(f32), device=device)
    torch_scale_b = torch.randint(1, 5, [num_group_N, num_group_K], dtype=dtype_to_torch(f32), device=device)

    hidet_a = hidet.from_torch(torch_a.to(dtype=dtype))
    hidet_b = hidet.from_torch(torch_b.to(dtype=dtype))
    hidet_scale_a = hidet.from_torch(torch_scale_a)
    hidet_scale_b = hidet.from_torch(torch_scale_b)

    hidet_result: hidet.Tensor = matmul_f8_scaled(
        hidet_a, hidet_b, scale_a=hidet_scale_a, scale_b=hidet_scale_b, output_dtype=output_dtype
    )

    # Verify with vLLM implementation
    # from vllm import _custom_ops as ops
    # vllm_result: torch.Tensor = ops.cutlass_scaled_mm(
    #     torch_a.to(dtype=dtype),
    #     torch_b.to(dtype=dtype).T,
    #     scale_a=torch_scale_a.T,
    #     scale_b=torch_scale_b.T ,
    #     out_dtype=dtype_to_torch(output_dtype)
    # )

    # from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_block_fp8_matmul
    # vllm_result = w8a8_block_fp8_matmul(
    #     torch_a.to(dtype=dtype),
    #     torch_b.to(dtype=dtype),
    #     torch_scale_a.T,
    #     torch_scale_b,
    #     block_size=[group_n,group_k],
    #     output_dtype=dtype_to_torch(output_dtype),
    # )

    for i in range(num_group_M):
        for j in range(num_group_K):
            torch_a[i * group_m : (i + 1) * group_m, j * group_k : (j + 1) * group_k] *= torch_scale_a[j, i]

    for i in range(num_group_N):
        for j in range(num_group_K):
            torch_b[i * group_n : (i + 1) * group_n, j * group_k : (j + 1) * group_k] *= torch_scale_b[i, j]

    torch_result = torch._scaled_mm(
        torch_a.to(dtype=dtype),
        torch_b.to(dtype=dtype).T,
        out_dtype=dtype_to_torch(output_dtype),
        scale_a=torch.tensor(1.0, device=device),
        scale_b=torch.tensor(1.0, device=device),
    )

    torch.testing.assert_close(actual=hidet_result.torch(), expected=torch_result, atol=atol, rtol=rtol)


@pytest.mark.requires_cuda
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
        a = torch.randint(low=-3, high=3, size=a_shape, dtype=dtype, device='cuda') / 5
        b = torch.randint(low=-3, high=3, size=b_shape, dtype=dtype, device='cuda')

        c_correct = torch.matmul(a, torch.transpose(b, 0, 1))
        ahi = hidet.from_torch(a)
        bhi = hidet.from_torch(b)
        ahi_symbol = hidet.symbol_like(ahi)
        bhi_symbol = hidet.symbol_like(bhi)
        cc = ops.matmul_nt(ahi_symbol, bhi_symbol)
        graph = hidet.graph.trace_from(cc, inputs=[ahi_symbol, bhi_symbol])
        graph_opt = hidet.graph.optimize(graph)
        c_hi = graph_opt(ahi, bhi)

        assert_torch_allclose(c_hi.cpu(), c_correct.cpu(), atol=1e-2, rtol=1e-2)


@pytest.mark.requires_cuda
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [[1, 125, 127], [127, 128]],
        [[1, 1024, 1024 + 8], [1024 + 8, 1024 - 8]],
        [[2, 1032, 1032], [1032, 1032]],
        [[1, 126, 128], [128]],
    ],
)
@pytest.mark.parametrize('transpose_b', [False, True])
def test_matmul_cublas(a_shape, b_shape, transpose_b):
    if transpose_b:
        b_shape = b_shape[::-1]
        torch_func = lambda x, y: torch.matmul(x, torch.transpose(y, 0, 1)) if len(b_shape) >= 2 else torch.matmul(x, y)
    else:
        torch_func = lambda x, y: torch.matmul(x, y)
    k_size = a_shape[-1]
    check_torch_binary(
        a_shape,
        b_shape,
        torch_func=torch_func,
        hidet_func=lambda x, y: ops.matmul_cublas(x, y, transpose_b=transpose_b),
        device='cuda',
        dtype='float16',
        atol=2e-2,
        rtol=2e-2,
        a_input_scale=1.0 / math.sqrt(k_size),
        b_input_scale=1.0 / math.sqrt(k_size),
    )


@pytest.mark.requires_cuda
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


@pytest.mark.requires_cuda_hopper
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
def test_matmul_bf16_sm90(a_shape, b_shape):
    from hidet.graph.ops.matmul.matmul_f16_sm90 import matmul_f16_sm90

    a = torch.randn(*a_shape, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(*b_shape, dtype=torch.bfloat16, device='cuda')
    c_correct = torch.matmul(a, b).to(dtype=torch.float32)
    ahi = hidet.from_torch(a)
    bhi = hidet.from_torch(b)
    ahi_symbol = hidet.symbol_like(ahi)
    bhi_symbol = hidet.symbol_like(bhi)
    cc = ops.squeeze(matmul_f16_sm90(ahi_symbol, bhi_symbol), 0)
    graph = hidet.graph.trace_from(cc, inputs=[ahi_symbol, bhi_symbol])
    graph_opt = hidet.graph.optimize(graph)
    c_hi = graph_opt(ahi, bhi).to(dtype='float32')
    np.testing.assert_allclose(c_hi.cpu().numpy(), c_correct.cpu().numpy(), atol=1e-1, rtol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__])
