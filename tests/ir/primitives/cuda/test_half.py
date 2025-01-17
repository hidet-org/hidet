import pytest
import hidet
import torch


@pytest.mark.requires_cuda
def test_sub_f16x2():
    from hidet.lang import attrs
    from hidet.lang.types import uint32
    from hidet.ir.primitives.cuda import sub_f16x2

    with hidet.script_module() as script_module:

        @hidet.script
        def sub_f16x2_test(c: ~uint32, a: uint32, b: uint32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 32

            sub_f16x2(c, a, b)

    kernel = script_module.build()

    c_int32 = torch.zeros([1], dtype=torch.int32, device='cuda')
    a = torch.asarray([3.0, 4.0], dtype=torch.float16, device='cuda')
    b = torch.asarray([1.0, 0.0], dtype=torch.float16, device='cuda')
    a_int32 = a.view(torch.int32).item()
    b_int32 = b.view(torch.int32).item()
    kernel(c_int32, a_int32, b_int32)
    c = c_int32.view(torch.float16)
    assert torch.allclose(c, a - b)


@pytest.mark.requires_cuda
def test_fma_f16x2():
    from hidet.lang import attrs
    from hidet.lang.types import uint32
    from hidet.ir.primitives.cuda import fma_f16x2

    with hidet.script_module() as script_module:

        @hidet.script
        def fma_f16x2_test(d: ~uint32, a: uint32, b: uint32, c: uint32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 32

            fma_f16x2(d, a, b, c)

    kernel = script_module.build()

    d_int32 = torch.zeros([1], dtype=torch.int32, device='cuda')
    a = torch.asarray([3.0, 4.0], dtype=torch.float16, device='cuda')
    b = torch.asarray([1.0, 5.0], dtype=torch.float16, device='cuda')
    c = torch.asarray([33.0, 44.0], dtype=torch.float16, device='cuda')
    a_int32 = a.view(torch.int32).item()
    b_int32 = b.view(torch.int32).item()
    c_int32 = c.view(torch.int32).item()
    kernel(d_int32, a_int32, b_int32, c_int32)
    d = d_int32.view(torch.float16)
    assert torch.allclose(d, a * b + c)
