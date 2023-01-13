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
from typing import List, Union
from ..utils import normalize_stride


def infer_conv2d_shape(
    x_shape: List[int], w_shape: List[int], strides: Union[int, List[int]], groups: int, dilations: List[int]
) -> List[int]:
    n, c, h, w = x_shape
    oc, gc, kx, ky = w_shape
    sx, sy = normalize_stride(strides)
    dilx, dily = dilations
    if gc * groups != c:
        msg = 'Conv2d: x has {} input channels, w has {} group channels, and groups={}'.format(c, gc, groups)
        raise ValueError(msg)
    if oc % groups != 0:
        msg = 'Conv2d expects out_channels % groups == 0, got out_channels {} and groups {}'.format(oc, groups)
        raise ValueError(msg)
    p, q = (h - dilx * (kx - 1) - 1) // sx + 1, (w - dily * (ky - 1) - 1) // sy + 1
    return [n, oc, p, q]
