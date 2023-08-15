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
from hidet.ir import primitives as prim
from hidet.ir.compute import reduce
from hidet.ir.expr import cast
from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.utils import compute, input_like, normalize_dim


class LpNormTask(Task):
    """
    Performs LP normalization along specified dimension.
    For a tensor input of shape [n0, n1, ..., ndim, ..., nk], each
    ndim-element vector v along dimension dim is transformed as:
    v = v / max(sum(pow(abs(v), p)), eps)
    """

    def __init__(self, x: TensorNode, p: float, dim: int, eps: float):
        x_shape = x.const_shape
        y_shape = x_shape
        dtype = x.type.dtype
        reduce_shape = []
        other_shape = []
        for idx, size in enumerate(x_shape):
            if idx == dim:
                reduce_shape.append(size)
            else:
                other_shape.append(size)

        def sum_compute(*indices):
            def sum_reduce(*reduction_axis):
                x_indices = []
                p = 0
                q = 0
                for i in range(len(x.shape)):
                    if i != dim:
                        x_indices.append(indices[p])
                        p += 1
                    else:
                        x_indices.append(reduction_axis[q])
                        q += 1
                assert p == len(indices) and q == len(reduction_axis)
                # Force fp32 reduction for accuracy
                return prim.pow(cast(prim.abs(x[x_indices]), 'float32'), p)

            return reduce(shape=reduce_shape, fcompute=sum_reduce, reduce_type='sum')

        sum_ = compute(name='sum', shape=other_shape, fcompute=sum_compute)

        p_norm = compute(name='p_norm', shape=other_shape, fcompute=lambda *indices: prim.pow(sum_[indices], 1.0 / p))

        def y_compute(*indices):
            norm_indices = [index for i, index in enumerate(indices) if i != dim]
            return cast(x[indices] / prim.max(p_norm[norm_indices], eps), dtype)

        y = compute(name='y', shape=y_shape, fcompute=y_compute)

        super().__init__(name='lp_norm', inputs=[x], outputs=[y], attributes={'p': p, 'dim': dim, 'eps': eps})


class LpNormOp(Operator):
    def __init__(self, x: Tensor, p: float, dim: int, eps: float):
        super().__init__(
            inputs=[x], attributes={'p': p, 'dim': dim, 'eps': eps}, task=LpNormTask(input_like(x, 'x'), p, dim, eps)
        )


def lp_norm(x: Tensor, p=2.0, dim=1, eps=1e-12):
    """LP norm.

    Parameters
    ----------
    x: Tensor
        The data to be normalized.
    p: float
        The exponent value in the norm formulation.
    dim: int
        The dimension to reduce.
    eps: float
        Small value to avoid division by zero.

    Returns
    -------
    ret: Tensor
        The normalized tensor.
    """
    # Normalize dim
    dim = normalize_dim(dim, rank=len(x.shape))
    return LpNormOp(x, p, dim, eps).outputs[0]
