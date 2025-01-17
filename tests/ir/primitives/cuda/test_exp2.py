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
def test_exp2():
    from hidet.lang import attrs
    from hidet.ir.primitives.math import exp2
    from hidet.ir.dtypes import f32
    from hidet.lang.cuda import threadIdx

    with hidet.script_module() as script_module:

        @hidet.script
        def func(out: f32[32]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 32
            attrs.cuda.grid_dim = 1

            t = threadIdx.x
            out[threadIdx.x] = exp2(f32(t))

    func = script_module.build()

    out = torch.empty((32,), dtype=torch.float32, device="cuda")
    out = hidet.from_torch(out)
    func(out)
    import numpy as np

    groundtruth = np.array([2**i for i in range(32)], dtype=np.float32)
    np.testing.assert_equal(out.cpu().numpy(), groundtruth)


@pytest.mark.requires_cuda
def test_exp2_f16():
    from hidet.lang import attrs
    from hidet.ir.primitives.math import exp2
    from hidet.ir.dtypes import f16
    from hidet.lang.cuda import threadIdx

    with hidet.script_module() as script_module:

        @hidet.script
        def func(out: f16[4]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 32
            attrs.cuda.grid_dim = 1

            t = threadIdx.x
            if t < 4:
                out[threadIdx.x] = exp2(f16(t))

    func = script_module.build()

    out = torch.empty((4,), dtype=torch.float16, device="cuda")
    out = hidet.from_torch(out)
    func(out)
    import numpy as np

    groundtruth = np.array([2**i for i in range(4)], dtype=np.float16)
    np.testing.assert_equal(out.cpu().numpy(), groundtruth)


if __name__ == "__main__":
    hidet.option.cache_dir("./exp2")
    hidet.option.save_lower_ir(True)

    pytest.main([__file__])
