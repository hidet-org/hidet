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
from .utils import Tensor, normalize_dim
from .arithmetic import square, rsqrt


def normalize(x: Tensor, dims: List[int], epsilon: float = 1e-5) -> Tensor:
    x = x - x.mean(dims, keep_dim=True)
    variance = square(x).mean(dims, keep_dim=True)
    return x * rsqrt(variance + epsilon)


def batch_norm_infer(x: Tensor, running_mean: Tensor, running_var: Tensor, epsilon=1e-5, axis=1) -> Tensor:
    rank = len(x.shape)
    axis = normalize_dim(axis, rank)

    assert len(running_mean.shape) == 1 and len(running_var.shape) == 1
    assert x.shape[axis] == running_mean.shape[0] == running_var.shape[0]

    running_mean = running_mean.unsqueeze([dim for dim in range(rank) if dim != axis])
    running_var = running_var.unsqueeze([dim for dim in range(rank) if dim != axis])
    return (x - running_mean) * rsqrt(running_var + epsilon)


def instance_norm(x: Tensor, axis: int = 1, epsilon: float = 1e-5) -> Tensor:
    """Instance norm.

    Parameters
    ----------
    x: Tensor
        The data to be normalized.
    axis: int
        The axis of channel dimension.
    epsilon: float
        The epsilon added to variance.

    Returns
    -------
    ret: Tensor
        The normalized tensor.
    """
    dims = [dim for dim in range(2, len(x.shape)) if dim != axis]
    return normalize(x, dims=dims, epsilon=epsilon)


def layer_norm(x: Tensor, num_last_dims: int = 1, epsilon: float = 1e-5) -> Tensor:
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

    Returns
    -------
    ret: Tensor
        The normalized tensor.
    """
    dims = list(range(len(x.shape) - num_last_dims, len(x.shape)))
    return normalize(x, dims=dims, epsilon=epsilon)
