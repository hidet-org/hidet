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
# pylint: disable=unused-import
from typing import Tuple, List, Union, Sequence, Optional
import builtins
from hidet.ir.layout import DataLayout
from hidet.ir.type import Int
from hidet.ir.expr import Var, Expr, Constant, is_constant
from hidet.ir.type import TensorType, tensor_type, DataType
from hidet.ir.task import Task, InverseMap
from hidet.ir.module import IRModule
from hidet.graph.operator import Operator, Tensor
from hidet.ir.compute import TensorNode, TensorInput, ReduceType, tensor_input, compute, reduce, arg_reduce
from hidet.ir.dtypes import int32


def input_like(tensor: Tensor, name: str) -> TensorInput:
    if not isinstance(tensor, Tensor):
        raise TypeError('Expect a hidet.Tensor, but got an object with type {}'.format(type(tensor)))
    return tensor_input(name, tensor.dtype, tensor.shape, tensor.layout)


def normalize_stride(stride: Union[Int, Sequence[Int]], dim=2) -> List[Int]:
    if isinstance(stride, (int, Expr)):
        return [stride for _ in range(dim)]
    elif isinstance(stride, (list, tuple)):
        if len(stride) == 1:
            return stride * dim
        elif len(stride) == dim:
            return stride
    msg = 'Stride must be an integer or a list of integer with length 1 or {}, but got {} of length {}'.format(
        dim, stride, len(stride)
    )
    raise ValueError(msg)


def normalize_dilations(dilations: Union[int, Sequence[int]], dim=2) -> List[int]:
    if isinstance(dilations, int):
        return [dilations for _ in range(dim)]
    elif isinstance(dilations, (list, tuple)):
        if len(dilations) == 1:
            return dilations * dim
        elif len(dilations) == dim:
            return dilations
    msg = 'Dilations must be an integer or a list of integer with length 1 or {}, but got {} of length {}'.format(
        dim, dilations, len(dilations)
    )
    raise ValueError(msg)


def normalize_kernel(kernel: Union[Int, Sequence[Int]], dim=2) -> List[Int]:
    if isinstance(kernel, (int, Expr)):
        return [kernel for _ in range(dim)]
    elif isinstance(kernel, (list, tuple)):
        if len(kernel) == 1:
            return list(kernel * dim)
        elif len(kernel) == dim:
            return list(kernel)
    msg = 'Kernel size must be an integer or a list of integer with length 1 or {}, but got {} of length {}'.format(
        dim, kernel, len(kernel)
    )
    raise ValueError(msg)


def normalize_output(output: Union[Int, Sequence[Int]], dim=2) -> List[Int]:
    # same as normalize_kernel
    return normalize_kernel(output, dim)


def normalize_padding(padding: Union[Int, Sequence[Int]], dim=2) -> List[Int]:
    if isinstance(padding, (int, Expr)):
        return [padding for _ in range(dim * 2)]
    elif isinstance(padding, (list, tuple)):
        if len(padding) == 1:
            return list(padding * (2 * dim))
        elif len(padding) == dim:
            return list(padding + padding)
        elif len(padding) == dim * 2:
            return list(padding)
    raise ValueError(
        'Padding must be an integer or a list of integer with length 1, '
        '{}, or {}, but got {}'.format(dim, dim * 2, padding)
    )


def normalize_dim(dim: Optional[Union[Int, Sequence[Int]]], rank: int) -> Union[Int, List[Int]]:
    """
    normalize a dim from [-rank, rank] or None to [0, rank].
    """
    if isinstance(dim, (list, tuple)):
        return [normalize_dim(d, rank) for d in dim]
    else:
        original_dim = dim
        if dim is None:
            dim = rank
        if is_constant(dim) and int(dim) < 0:
            dim += rank
        if is_constant(dim) and not 0 <= int(dim) <= rank:
            raise ValueError('Given dim {} is not a valid dim for rank {}'.format(original_dim, rank))
        return dim


def normalize_index(index: Optional[int], dim_size, default) -> int:
    """
    normalize an index from [-oo, oo] or None to [0, dim_size]
    """
    if index is None:
        return default
    elif index < 0:
        return max(index + dim_size, 0)
    elif 0 <= index <= dim_size:
        return index
    else:
        return dim_size


def is_contiguous_norm(dims, rank):
    dims = normalize_dim(dims, rank)
    return max(dims) - min(dims) == len(dims) - 1 and max(dims) == rank - 1


def resolve_out_dtype(input_dtypes: Sequence[Union[DataType, str]]) -> str:
    from hidet.ir.utils.type_utils import numeric_promotion

    if len(input_dtypes) == 0:
        raise ValueError('Expect at least one input dtype to resolve the output dtype.')
    out_dtype = input_dtypes[0]
    for input_dtype in input_dtypes[1:]:
        out_dtype = numeric_promotion(out_dtype, input_dtype)
    return out_dtype.name


def can_broadcast(src_shape: Sequence[Int], dst_shape: Sequence[Int]) -> bool:
    if len(dst_shape) < len(src_shape):
        return False
    src_shape = [1 for _ in range(len(dst_shape) - len(src_shape))] + list(src_shape)
    for a, b in zip(src_shape, dst_shape):
        if is_constant(a, b) and a not in [1, b]:
            return False
    return True


def can_mutually_broadcast(x_shape: Sequence[Int], y_shape: Sequence[Int]) -> bool:
    x_shape, y_shape = list(x_shape), list(y_shape)
    while len(x_shape) < len(y_shape):
        x_shape = [1] + x_shape
    while len(y_shape) < len(x_shape):
        y_shape = [1] + y_shape
    return all(p == q or p == 1 or q == 1 for p, q in zip(x_shape, y_shape) if is_constant(p, q))


def broadcast_shape(x_shape: Sequence[Int], y_shape: Sequence[Int]) -> List[Int]:
    """
    Broadcast two shapes with the same rule as numpy.
    Please refer to https://numpy.org/doc/stable/user/basics.broadcasting.html for details.
    """
    x_shape, y_shape = list(x_shape), list(y_shape)
    orig_shapes = x_shape, y_shape
    while len(x_shape) < len(y_shape):
        x_shape = [int32(1)] + x_shape
    while len(y_shape) < len(x_shape):
        y_shape = [int32(1)] + y_shape
    result_shape = []
    for p, q in zip(x_shape, y_shape):
        if is_constant(p) and p == 1:
            result_shape.append(q)
        elif is_constant(q) and q == 1:
            result_shape.append(p)
        elif is_constant(p, q):
            if p != q:
                raise ValueError(
                    'can not broadcast two arrays with shape {} and {}'.format(orig_shapes[0], orig_shapes[1])
                )
            result_shape.append(p)
        else:
            result_shape.append(p if is_constant(p) else q)
    return result_shape


def broadcast_shapes(shapes: Sequence[Sequence[Int]]) -> List[Int]:
    assert len(shapes) >= 1
    expanded_shape = list(shapes[0])
    for shape in shapes:
        expanded_shape = broadcast_shape(expanded_shape, shape)
    return expanded_shape


def broadcast_indices(indices: Sequence[Int], shape: Sequence[Int], out_shape: Sequence[Int]) -> List[Expr]:
    if len(indices) != len(out_shape):
        raise ValueError('Number of indices {} does not match the output shape {}'.format(indices, out_shape))

    pad_dim = len(out_shape) - len(shape)
    indices = list(indices[pad_dim:])
    for idx, dim in enumerate(shape):
        if is_constant(dim) and dim == 1:
            indices[idx] = 0
    return indices


def convert_to_tensor(value: Union[int, float, bool, complex, Tensor], involved_tensor: Tensor) -> Tensor:
    from hidet.graph.tensor import full_like

    if isinstance(value, Tensor):
        return value

    if involved_tensor.dtype.is_complex():
        return full_like(involved_tensor, fill_value=value, shape=[], dtype=involved_tensor.dtype)
    elif involved_tensor.dtype.is_float():
        if isinstance(value, (bool, int, float)):
            return full_like(involved_tensor, fill_value=value, shape=[])
        else:
            return full_like(involved_tensor, fill_value=value, shape=[], dtype='complex64')
    elif involved_tensor.dtype.is_integer():
        if isinstance(value, (bool, int)):
            return full_like(involved_tensor, fill_value=value, shape=[])
        else:
            return full_like(involved_tensor, fill_value=value, shape=[], dtype='float32')
    else:
        raise ValueError('Can not recognize dtype {}'.format(involved_tensor.dtype))
