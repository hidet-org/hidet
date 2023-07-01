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
from hidet.ir.compute import reduce
from hidet.graph.ops.utils import TensorNode
from hidet.graph.ops.utils import compute


def reduce_shape(x: TensorNode, dims: List[int], keep_dim: bool):
    y_shape = []
    dims = [i if i >= 0 else len(x.shape) + i for i in dims]
    for i in range(len(x.shape)):
        if i in dims:
            if keep_dim:
                y_shape.append(1)
        else:
            y_shape.append(x.shape[i])
    return y_shape


def reduce(x: TensorNode, dims: List[int], keep_dim: bool, reduce_type: str, accumulate_dtype: str = 'float32'):
    y_shape = reduce_shape(x, dims, keep_dim)

    def fcompute(*indices):
        def reduce_fcompute(*reduce_indices):
            x_indices = []
            p = 0
            q = 0
            for i in range(len(x.shape)):
                if i not in dims:
                    x_indices.append(indices[p])
                    p += 1
                else:
                    x_indices.append(reduce_indices[q])
                    q += 1
                    if keep_dim:
                        p += 1
            assert p == len(indices) and q == len(reduce_indices)
            return x[x_indices]

        y_shape = [x.shape[i] for i in dims]
        return reduce(
            shape=y_shape, fcompute=reduce_fcompute, reduce_type=reduce_type, accumulate_dtype=accumulate_dtype
        )

    y = compute(name='y', shape=y_shape, fcompute=fcompute)
    return y
