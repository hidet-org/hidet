import torch

import hidet
from hidet.lang import script, attrs, asm, deref
from hidet.lang.types import int32


def test_hidet_script_assign_list():
    with hidet.script_module() as script_module:

        @script
        def cuda_load(addr: ~int32, v0: ~int32):
            attrs.func_kind = "cuda_internal"
            attrs.func_name = 'cuda_load'
            template = "ld.b32 %0, [%1];"
            outputs = [deref(v0)]
            asm(template, outputs=outputs, inputs=[addr], is_volatile=True)

        @script
        def kernel(src: ~int32, dst: ~int32):
            attrs.func_kind = "cuda_kernel"
            attrs.func_name = 'kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 1

            reg_int32: int32  # define a variable without initialization
            cuda_load(src, ~reg_int32)  # load the value from the global address
            dst[0] = reg_int32

    built = script_module.build()

    src = torch.asarray([123], dtype=torch.int32, device='cuda')
    dst = torch.zeros(1, dtype=torch.int32, device='cuda')
    built(src, dst)
    torch.testing.assert_close(src, dst)
