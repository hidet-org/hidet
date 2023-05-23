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
from hidet.testing.torch_utils import check_module, FunctionalModule


@pytest.mark.parametrize('shape', [[1, 16, 1024, 128], [4, 4, 4096, 64]])
@pytest.mark.parametrize('attn_mask_type', [None, 'bool', 'float16', 'causal'])
def test_sdpa(shape, attn_mask_type):
    q = torch.randn(shape, dtype=torch.float16)
    k = torch.randn(shape, dtype=torch.float16)
    v = torch.randn(shape, dtype=torch.float16)
    is_causal = False
    attn_mask = None
    mask_shape = q.shape[:-2] + (q.shape[-2], q.shape[-2])
    if attn_mask_type == 'causal':
        is_causal = True
    elif attn_mask_type == 'bool':
        attn_mask = torch.rand(mask_shape) > 0.5
    elif attn_mask_type == 'float16':
        attn_mask = torch.randn(mask_shape, dtype=torch.float16)

    check_module(
        FunctionalModule(
            op=lambda _q, _k, _v, _attn_mask, _is_causal: torch.nn.functional.scaled_dot_product_attention(
                _q, _k, _v, attn_mask=_attn_mask, is_causal=_is_causal
            )
        ),
        [q, k, v, attn_mask, is_causal],
        atol=1e-2,
        rtol=1e-2,
    )


if __name__ == '__main__':
    pytest.main([__file__])
