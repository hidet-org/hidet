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
import pytest
import torch
import numpy as np

hidet.option.save_lower_ir()


def test_profile_config():
    a = hidet.randn([1, 10, 10], device='cuda')
    b = hidet.randn([1, 10, 10], device='cuda')
    hidet.option.search_space(1)
    hidet.option.bench_config(1, 1, 1)
    c = hidet.ops.batch_matmul(a, b)
    hidet.option.search_space(0)


@pytest.mark.parametrize(
    "equation, operand_shapes",
    [
        ["bhwc,hkc->bhwk", [[400, 14, 14, 80], [14, 14, 80]]],
        ["bhwc,wkc->bhwk", [[400, 14, 14, 80], [14, 14, 80]]],
        ["bhwc,wkc->bhwk", [[16, 64, 64, 80], [64, 64, 80]]],
        ["bhwc,hkc->bhwk", [[16, 64, 64, 80], [64, 64, 80]]],
        ["ijk,k->ij", [[20, 30, 40], [40]]],
        ["abc,c->ab", [[10, 20, 30], [30]]],
        ["abcd,cd->ab", [[10, 20, 30, 40], [30, 40]]],
        ["ij,j->i", [[10, 20], [20]]],
        ["abc,bc->a", [[10, 20, 30], [20, 30]]],
    ],
)
def test_einsum(equation, operand_shapes):
    operands_torch = [torch.randn(shape) for shape in operand_shapes]
    operands_hidet = [hidet.from_torch(op) for op in operands_torch]
    result_torch = torch.einsum(equation, *operands_torch)
    result_hidet = hidet.ops.einsum(equation, operands_hidet)

    result_torch = result_torch.cpu().numpy()
    result_hidet = result_hidet.cpu().numpy()

    assert result_torch.shape == result_hidet.shape, "Output shape mismatch in einsum"

    np.testing.assert_allclose(result_torch, result_hidet, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
    pytest.main(__file__)
