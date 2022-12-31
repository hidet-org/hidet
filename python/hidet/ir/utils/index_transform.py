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
from typing import List
from ..expr import Expr, convert


def index_serialize(indices: List[Expr], shape: List[int]) -> Expr:
    if len(shape) == 0:
        return convert(0)
    scalar_index: Expr = convert(0)
    acc = 1
    for idx_value, extent in reversed(list(zip(indices, shape))):
        scalar_index += idx_value * acc
        acc *= extent
    return scalar_index


def index_deserialize(scalar_index: Expr, shape: List[int]) -> List[Expr]:
    if len(shape) == 0:
        return []
    indices = []
    acc = 1
    for r, extent in enumerate(reversed(shape)):
        if r < len(shape) - 1:
            indices.append(scalar_index // acc % extent)
        else:
            indices.append(scalar_index // acc)
        acc *= extent
    return list(reversed(indices))
