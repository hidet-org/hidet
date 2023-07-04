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
from hidet.ir.expr import if_then_else, logical_and, convert
from hidet.ir.compute.primitives import TensorNode, compute


def pad(data: TensorNode, pads: List[int], value: float):
    shape = data.shape
    rank = len(shape)
    assert rank * 2 == len(pads)
    out_shape = [a + b + c for a, b, c in zip(pads[:rank], shape, pads[rank:])]

    value = convert(value, dtype=data.type.dtype.name)

    def fmap(*indices):
        indices = [idx - beg for idx, beg in zip(indices, pads[:rank])]
        cond = logical_and(*[logical_and(0 <= idx, idx < shape[i]) for i, idx in enumerate(indices)])
        return if_then_else(cond, data[indices], value)

    out = compute('out', shape=out_shape, fcompute=fmap)
    return out
