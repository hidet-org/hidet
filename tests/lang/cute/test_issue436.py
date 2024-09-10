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
from hidet.testing.torch_utils import Backend

import torch


def test_gpt2xl_issue436():
    backend = Backend('hidet', 'max-autotune', 'float16', '/gpt2.cache')

    causal_mask = torch.tril(torch.zeros((50, 50), dtype=torch.bool, device='cuda')).view(1, 1, 50, 50)
    mask_value = torch.full([], 10, dtype=torch.float16, device='cuda')

    def attn(query, key):
        attn_weights = torch.matmul(query, key)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        return attn_weights

    # hidet.option.cache_dir("./issue436")
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    q = torch.randn(1, 25, 50, 64, dtype=torch.float16, device='cuda')
    k = torch.randn(1, 25, 64, 50, dtype=torch.float16, device='cuda')
    model = attn

    with torch.inference_mode(True):
        t = model(q, k)
        model = backend.compile(model)
        h = model(q, k)

        import numpy as np

        np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
        np.testing.assert_allclose(actual=t.cpu().numpy(), desired=h.cpu().numpy(), rtol=1e-2)


if __name__ == "__main__":
    import pytest

    pytest.main(__file__)
