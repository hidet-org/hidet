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

from ..utils import Tensor, normalize_dim
from ..arithmetic import rsqrt
from .norm import normalize
from .norm_f16 import normalize_f16


def resolve_norm_func(dtype):
    if dtype == hidet.float32:
        return normalize
    elif dtype == hidet.float16:
        return normalize_f16
    else:
        raise NotImplementedError("normalize function for dtype {} is not implemented".format(dtype))


def batch_norm_infer(x: Tensor, running_mean: Tensor, running_var: Tensor, epsilon=1e-5, axis=1) -> Tensor:
    rank = len(x.shape)
    axis = normalize_dim(axis, rank)

    assert len(running_mean.shape) == 1 and len(running_var.shape) == 1
    assert x.shape[axis] == running_mean.shape[0] == running_var.shape[0]

    running_mean = running_mean.unsqueeze([dim for dim in range(rank) if dim != axis])
    running_var = running_var.unsqueeze([dim for dim in range(rank) if dim != axis])
    return (x - running_mean) * rsqrt(running_var + epsilon)


def instance_norm(x: Tensor, epsilon: float = 1e-5, accumulate_dtype: str = 'float32') -> Tensor:
    """Instance norm.

    Parameters
    ----------
    x: Tensor
        The data to be normalized.
    axis: int
        The axis of channel dimension.
    epsilon: float
        The epsilon added to variance.
    accumulate_dtype: str
        The precision used for accumulation during reduction

    Returns
    -------
    ret: Tensor
        The normalized tensor.
    """
    dims = [dim for dim in range(2, len(x.shape))]
    norm_func = resolve_norm_func(x.dtype)
    return norm_func(x, axis=dims, epsilon=epsilon, accumulate_dtype=accumulate_dtype)


def layer_norm(x: Tensor, num_last_dims: int = 1, epsilon: float = 1e-5, accumulate_dtype: str = 'float32') -> Tensor:
    """
    Layer norm.

    Parameters
    ----------
    x: Tensor
        The data to be normalized.
    num_last_dims: int
        The number of dimensions to be normalized, starting from the end dimension of x.
    epsilon: float
        The epsilon added to variance.
    accumulate_dtype: str
        The precision used for accumulation during reduction

    Returns
    -------
    ret: Tensor
        The normalized tensor.
    """
    norm_func = resolve_norm_func(x.dtype)
    dims = list(range(len(x.shape) - num_last_dims, len(x.shape)))
    return norm_func(x, axis=dims, epsilon=epsilon, accumulate_dtype=accumulate_dtype)


def group_norm(x: Tensor, num_groups, epsilon: float = 1e-5, accumulate_dtype: str = 'float32'):
    """
    Group norm.

    Parameters
    ----------
    x: Tensor
        The data to be normalized.
    num_groups: int
        The number of groups
    epsilon: float
        The epsilon added to variance.
    accumulate_dtype: str
        The precision used for accumulation during reduction

    Returns
    -------
    ret: Tensor
        The normalized tensor.
    """
    # first split out the group dimension
    x_shape = list(x.shape)
    new_shape = x_shape[:]
    grouped_rank = 1
    grouped_dim = new_shape[grouped_rank]
    assert grouped_dim % num_groups == 0

    new_shape[grouped_rank] = int(grouped_dim // num_groups)
    new_shape.insert(grouped_rank, num_groups)

    x = x.reshape(new_shape)
    dims = list(range(2, len(x.shape)))
    norm_func = resolve_norm_func(x.dtype)
    normed = norm_func(x, axis=dims, epsilon=epsilon, accumulate_dtype=accumulate_dtype)

    return normed.reshape(x_shape)
