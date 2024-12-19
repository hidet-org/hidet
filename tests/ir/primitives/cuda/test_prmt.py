import torch
import hidet


def test_prmt():
    from hidet.lang import attrs, script
    from hidet.lang.types import uint32
    from hidet.ir.primitives.cuda import prmt

    with hidet.script_module() as script_module:

        @script
        def kernel(d_ptr: ~uint32, a: uint32, b: uint32, c: uint32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 32

            prmt(d=d_ptr, a=a, b=b, c=c)

    func = script_module.build()

    d_int32 = torch.empty([1], dtype=torch.int32, device='cuda')
    func(d_int32, 0x00000201, 0x00000064, 0x4140)
    d_int32 = d_int32.item()
    assert d_int32 == 0x64026401
