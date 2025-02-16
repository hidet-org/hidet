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
# %%
import numpy as np
import pytest


import hidet
from hidet.ir import data_type
from hidet.ir.primitives.hip.mfma import MfmaConfig, mfma_sync
from hidet.lang.hip import register_tensor, threadIdx
from hidet.lang import attrs

should_run = hidet.hip.available()
if should_run:
    capability = hidet.hip.capability()
    should_run = capability.gcnArchName == 'gfx90a'


@pytest.mark.skipif(not should_run, reason='hip is not available or gcnArch is not gfx90a')
@pytest.mark.parametrize(
    'config',
    [
        MfmaConfig.v_mfma_f32_16x16x16f16(),
        MfmaConfig.v_mfma_f32_32x32x8f16(),
        MfmaConfig.v_mfma_f32_16x16x4f32(),
        MfmaConfig.v_mfma_f32_32x32x2f32(),
    ],
)
def test_mfma_tensor_core(config: MfmaConfig):
    ab_dtype = data_type(config.input_dtype)
    c_dtype = data_type(config.output_dtype)

    with hidet.script_module() as module:

        @hidet.script
        def test_matmul(
            a: ab_dtype[config.m, config.k], b: ab_dtype[config.k, config.n], c: c_dtype[config.m, config.n]
        ):
            attrs.hip.block_dim = 64
            attrs.hip.grid_dim = 1

            a_regs = register_tensor(ab_dtype, [config.a_elements])
            b_regs = register_tensor(ab_dtype, [config.b_elements])
            c_regs = register_tensor(c_dtype, [config.c_elements])
            for p in range(config.c_elements):
                c_regs[p] = 0.0

            p = 0
            for i, k in config.a_load_map.on(threadIdx.x):
                a_regs[p] = a[i, k]
                p += 1

            p = 0
            for k, j in config.b_load_map.on(threadIdx.x):
                b_regs[p] = b[k, j]
                p += 1

            mfma_sync(config, a_regs, b_regs, c_regs)

            p = 0
            for i, j in config.c_store_map.on(threadIdx.x):
                c[i, j] = c_regs[p]
                p += 1

    ir_module = module.ir_module()
    cmodule = ir_module.build()

    a = hidet.randn([config.m, config.k], dtype=config.input_dtype, device='hip')
    b = hidet.randn([config.k, config.n], dtype=config.input_dtype, device='hip')
    c = hidet.empty([config.m, config.n], dtype=config.output_dtype, device='hip')

    cmodule(a, b, c)
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    c = c.cpu().numpy()

    cn = np.matmul(a, b)

    if config.input_dtype == 'f16':
        assert np.allclose(c, cn, atol=1e-2)
    else:
        assert np.allclose(c, cn, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
