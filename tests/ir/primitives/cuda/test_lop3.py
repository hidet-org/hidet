import pytest
import torch
import hidet


@pytest.mark.requires_cuda
def test_lop3():
    from hidet.lang import attrs, script
    from hidet.lang.types import uint32
    from hidet.ir.primitives.cuda import lop3

    with hidet.script_module() as script_module:

        @script
        def kernel(d_ptr: ~uint32, a: uint32, b: uint32, c: uint32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 32

            lop3(d_ptr, a, b, c, imm_lut=(0xF0 & 0xCC) | 0xAA)

    func = script_module.build()

    d = torch.empty([1], dtype=torch.int32, device='cuda')
    func(d, 0xFFFFFFFF, 0x00FF00FF, 0x0E00EE00)
    assert d[0] == 0x0EFFEEFF
