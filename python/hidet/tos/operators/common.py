from typing import Tuple, List, Union, Sequence, Optional
from hidet.ir.layout import DataLayout
from hidet.ir.type import TensorType, tensor_type
from hidet.ir.task import Task, Grid
from hidet.tos.operator import Operator, Tensor
from hidet.ir.dialects.compute import TensorInput, tensor_input, compute, reduce, custom_compute

from hidet.ir.functors import inline_compute


def input_like(tensor: Tensor, name: str) -> TensorInput:
    # todo: make scope and device consistent
    device2scope = {
        'cuda': 'global',
        'cpu': 'host'
    }
    return tensor_input(name, tensor.dtype, tensor.shape, device2scope[tensor.device], tensor.layout)


def normalize_stride(stride: Union[int, Sequence[int]], dim=2) -> List[int]:
    if isinstance(stride, int):
        return [stride for _ in range(dim)]
    elif isinstance(stride, (list, tuple)):
        if len(stride) == 1:
            return stride * dim
        elif len(stride) == dim:
            return stride
    raise ValueError('Stride must be an integer or a list of integer with length 1 or {}, but got {}'.format(dim, stride))


def normalize_kernel(kernel: Union[int, Sequence[int]], dim=2) -> List[int]:
    if isinstance(kernel, int):
        return [kernel for _ in range(dim)]
    elif isinstance(kernel, (list, tuple)):
        if len(kernel) == 1:
            return kernel * dim
        elif len(kernel) == dim:
            return kernel
    raise ValueError('Kernel size must be an integer or a list of integer with length 1 or {}, but got {}'.format(dim, kernel))


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
    raise ValueError('Padding must be an integer or a list of integer with length 1, {}, or {}, but got {}'.format(dim, dim * 2, padding))


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
        if not (0 <= dim <= rank):
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

