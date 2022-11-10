# pylint: disable=unused-import
from typing import Tuple, List, Union, Sequence, Optional
import builtins
from hidet.ir.layout import DataLayout
from hidet.ir.expr import Var
from hidet.ir.type import TensorType, tensor_type, DataType
from hidet.ir.task import Task, InverseMap
from hidet.ir.func import IRModule
from hidet.graph.operator import Operator, Tensor
from hidet.ir.compute import TensorNode, tensor_input, compute, reduce, arg_reduce


def input_like(tensor: Tensor, name: str) -> TensorNode:
    return tensor_input(name, tensor.dtype, tensor.shape, tensor.layout)


def normalize_stride(stride: Union[int, Sequence[int]], dim=2) -> List[int]:
    if isinstance(stride, int):
        return [stride for _ in range(dim)]
    elif isinstance(stride, (list, tuple)):
        if len(stride) == 1:
            return stride * dim
        elif len(stride) == dim:
            return stride
    msg = 'Stride must be an integer or a list of integer with length 1 or {}, but got {}'.format(dim, stride)
    raise ValueError(msg)


def normalize_kernel(kernel: Union[int, Sequence[int]], dim=2) -> List[int]:
    if isinstance(kernel, int):
        return [kernel for _ in range(dim)]
    elif isinstance(kernel, (list, tuple)):
        if len(kernel) == 1:
            return kernel * dim
        elif len(kernel) == dim:
            return kernel
    msg = 'Kernel size must be an integer or a list of integer with length 1 or {}, but got {}'.format(dim, kernel)
    raise ValueError(msg)


def normalize_padding(padding: Union[int, Sequence[int]], dim=2) -> List[int]:
    if isinstance(padding, int):
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


def normalize_dim(dim: Optional[Union[int, Sequence[int]]], rank: int) -> Union[int, List[int]]:
    """
    normalize a dim from [-rank, rank] or None to [0, rank].
    """
    if isinstance(dim, (list, tuple)):
        return [normalize_dim(d, rank) for d in dim]
    else:
        original_dim = dim
        if dim is None:
            dim = rank
        if dim < 0:
            dim += rank
        if not 0 <= dim <= rank:
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


def resolve_out_dtype(input_dtypes: List[Union[DataType, str]]) -> str:
    from hidet.ir.utils.type_utils import numeric_promotation

    if len(input_dtypes) == 0:
        raise ValueError('Expect at least one input dtype to resolve the output dtype.')
    out_dtype = input_dtypes[0]
    for input_dtype in input_dtypes[1:]:
        out_dtype = numeric_promotation(out_dtype, input_dtype)
    return out_dtype.name


def broadcast_shape(x_shape: List[int], y_shape: List[int]) -> List[int]:
    """
    Broadcast two shapes with the same rule as numpy.
    Please refer to https://numpy.org/doc/stable/user/basics.broadcasting.html for details.
    """
    orig_shapes = x_shape, y_shape
    while len(x_shape) < len(y_shape):
        x_shape = [1] + x_shape
    while len(y_shape) < len(x_shape):
        y_shape = [1] + y_shape
    result_shape = []
    for p, q in zip(x_shape, y_shape):
        if p != q and p != 1 and q != 1:
            raise ValueError('can not broadcast two arrays with shape {} and {}'.format(orig_shapes[0], orig_shapes[1]))
        result_shape.append(builtins.max(p, q))
    return result_shape


def broadcast_shapes(shapes: List[List[int]]) -> List[int]:
    assert len(shapes) >= 1
    expanded_shape = shapes[0]
    for shape in shapes:
        expanded_shape = broadcast_shape(expanded_shape, shape)
    return expanded_shape


def broadcast_indices(indices, shape, out_shape):
    # used to support broadcast
    pad_dim = len(out_shape) - len(shape)
    indices = list(indices[pad_dim:])
    for idx, dim in enumerate(shape):
        if int(dim) == 1:
            indices[idx] = 0
    return indices
