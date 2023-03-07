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

from hidet.testing.onnx_utils import check_onnx_and_hidet


class SliceModule(torch.nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def forward(self, x):
        return x[self.indices]


@pytest.mark.parametrize('shape,indices', [((100,), slice(2, None))])
def test_slice(shape, indices):
    check_onnx_and_hidet(SliceModule(indices), [torch.randn(shape)])
