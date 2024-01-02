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
    print(data.cpu().numpy())


def test_int_2bit():
    from hidet.ir.dtypes import i4, u4, i4x8, i8, f16, f32
    from hidet.ir.expr import constant, cast
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory

    with hidet.script_module() as script_module:

        @hidet.script
        def func(out: f32[4, 4]):
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
            e = i8(b[3, 0])
            s1 = shared_tensor('float32', shape=[64, 64])
            s1[3, 4] = f16(b[4, 0])

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
                        d = i4(f32(d) + 1)
                        out[i, j] = f32(d)

    func = script_module.build()
    import torch
    data = torch.empty((4, 4), dtype=torch.float32, device="cuda")
    data = hidet.from_torch(data)
    func(data)
    print(data.cpu().numpy())



if __name__ == "__main__":
    hidet.option.cache_dir("./demo_int_subbyte")
    hidet.option.save_lower_ir(True)

    test_int_4bit()
