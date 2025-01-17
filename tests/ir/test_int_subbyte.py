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
import pytest

import torch
import hidet


@pytest.mark.requires_cuda
def test_int_4bit():
    from hidet.ir.dtypes import i4, u4, i4x8, i8, f16, f32
    from hidet.ir.expr import constant, cast
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory

    with hidet.script_module() as script_module:

        @hidet.script
        def func(out: f32[4, 4], inp: i4[4, 4]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 4

            a = constant(1, i4)
            b = register_tensor('int4b', shape=[8, 2])
            ptr = ~b[3, 1]
            ptr = ptr + 4
            ptr = ptr + (threadIdx.x * 444 + blockIdx.x * 888 + 555)
            c = register_tensor('i4x8', shape=[1])
            b[0, 1] = a
            b[0, 1] = b[0, 2]
            d = b[0, 1]
            s = shared_tensor('uint4b', shape=[7, 8])
            e = f32(s[2, 4])
            s1 = shared_tensor('float32', shape=[64, 64])
            s1[3, 4] = f16(b[4, 0])
            s2 = s[:, 4:]
            f = f32(s2[2, 0])

            data = register_tensor('int4b', shape=[4, 4])

            for i in range(4):
                for j in range(4):
                    if i == 0 and j == 0:
                        data[i, j] = i4(-8)
                    elif j == 0:
                        data[i, j] = i4(f32(data[i - 1, 3]) + 1)
                    else:
                        data[i, j] = i4(f32(data[i, j - 1]) + 1)

            if threadIdx.x == 0 and blockIdx.x == 0:
                for i in range(4):
                    for j in range(4):
                        d = data[i, j]
                        out[i, j] = f32(d)

    func = script_module.build()
    import torch

    data = torch.empty((4, 4), dtype=torch.float32, device="cuda")
    data = hidet.from_torch(data)
    inp = torch.empty((4, 2), dtype=torch.int8, device="cuda")
    inp = hidet.from_torch(inp)
    func(data, inp)
    import numpy as np

    groundtruth = np.resize(np.arange(-8, 8), (4, 4)).astype(np.float32)
    np.testing.assert_equal(data.cpu().numpy(), groundtruth)


@pytest.mark.requires_cuda
def test_int_2bit():
    from hidet.ir.dtypes import i2, u2, i8, f16, f32
    from hidet.ir.expr import constant, cast
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory

    with hidet.script_module() as script_module:

        @hidet.script
        def func(out: f32[2, 2]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 4

            a = constant(1, i2)

            data = register_tensor('int2b', shape=[2, 2])

            for i in range(2):
                for j in range(2):
                    if i == 0 and j == 0:
                        data[i, j] = i2(-2)
                    elif j == 0:
                        data[i, j] = i2(f32(data[i - 1, 1]) + 1)
                    else:
                        data[i, j] = i2(f32(data[i, j - 1]) + 1)

            if threadIdx.x == 0 and blockIdx.x == 0:
                for i in range(2):
                    for j in range(2):
                        d = data[i, j]
                        out[i, j] = f32(d)

    func = script_module.build()
    import torch

    data = torch.empty((2, 2), dtype=torch.float32, device="cuda")
    data = hidet.from_torch(data)
    func(data)
    import numpy as np

    groundtruth = np.resize(np.arange(-2, 2), (2, 2)).astype(np.float32)
    np.testing.assert_equal(data.cpu().numpy(), groundtruth)


@pytest.mark.requires_cuda
def test_write_int4_to_global_memory():
    from hidet.lang import attrs
    from hidet.lang.cuda import threadIdx
    from hidet.ir.dtypes import int4b, uint4b

    with hidet.script_module() as script_module:

        @hidet.script
        def kernel(d: ~uint4b):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 32

            i = threadIdx.x
            if i < 8:
                d[i] = uint4b(i % 0xF)

    func = script_module.build()
    d_int32 = torch.empty([1], dtype=torch.int32, device='cuda')
    func(d_int32)
    assert d_int32.item() == 0x76543210


if __name__ == "__main__":
    hidet.option.cache_dir("./demo_int_subbyte")
    hidet.option.save_lower_ir(True)

    pytest.main(__file__)
