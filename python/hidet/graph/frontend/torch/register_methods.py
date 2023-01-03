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
from __future__ import annotations
from typing import List, Union
import torch
from hidet.ir.type import DataType
from hidet.graph.tensor import Tensor
from hidet.graph import ops
from .interpreter import register_method
from .utils import dtype_from_torch, device_from_torch, dtype_to_torch


@register_method(torch.Tensor.cuda)
def tensor_cuda(self: Tensor) -> Tensor:
    if self.is_symbolic():
        raise NotImplementedError('hidet: torch.Tensor.cuda() is not supported for symbolic tensors.')
    return self.cuda()


@register_method(torch.Tensor.cpu)
def tensor_cpu(self: Tensor) -> Tensor:
    if self.is_symbolic():
        raise NotImplementedError('hidet: torch.Tensor.cpu() is not supported for symbolic tensors.')
    return self.cpu()


@register_method(torch.Tensor.to)
def tensor_to(self: Tensor, *args, **kwargs) -> Tensor:
    """
    There are three argument format for torch.Tensor.to:

    1. to(self, dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
    2. to(self, device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
    3. to(self, non_blocking=False, copy=False, memory_format=torch.preserve_format)

    The current implementation should work for most use cases.

    See Also
        https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
    """
    dtype = kwargs.get('dtype', None)
    device = kwargs.get('device', None)
    non_blocking = kwargs.get('non_blocking', False)
    copy = kwargs.get('copy', False)
    memory_format = kwargs.get('memory_format', torch.preserve_format)

    for arg in args:
        if isinstance(arg, torch.dtype):
            dtype = arg
        elif isinstance(arg, torch.device):
            if self.is_symbolic():
                raise NotImplementedError('hidet: Tensor.to(..., device=...) is not supported for symbolic tensors.')
            device = arg
        else:
            raise ValueError(f'Unsupported argument type: {type(arg)}')

    if memory_format != torch.preserve_format:
        raise NotImplementedError('hidet: torch.Tensor.to(..., memory_format=..., ...) is not supported.')

    _ = copy
    _ = non_blocking

    return self.to(
        dtype=dtype_from_torch(dtype).name if dtype else None, device=device_from_torch(device) if device else None
    )


@register_method(torch.Tensor.view)
def tensor_view(self: Tensor, *args) -> Tensor:
    if len(args) == 1 and isinstance(args[0], torch.dtype):
        if self.is_symbolic():
            raise NotImplementedError('hidet: torch.Tensor.view(dtype) is not supported for symbolic tensors for now.')
        assert len(args) == 1
        src_dtype: DataType = self.dtype
        dst_dtype: DataType = dtype_from_torch(args[0])
        if src_dtype == dst_dtype:
            return self

        if src_dtype.nbytes == dst_dtype.nbytes:
            new_shape = self.shape
        elif src_dtype.nbytes > dst_dtype.nbytes:
            assert src_dtype.nbytes % dst_dtype.nbytes == 0
            new_shape = self.shape[:-1] + tuple([self.shape[-1] * (src_dtype.nbytes // dst_dtype.nbytes)])
        elif src_dtype.nbytes < dst_dtype.nbytes:
            assert dst_dtype.nbytes % src_dtype.nbytes == 0  # these should already have been checked by pytorch
            assert self.shape[-1] % (dst_dtype.nbytes // src_dtype.nbytes) == 0
            new_shape = self.shape[:-1] + tuple([self.shape[-1] // (dst_dtype.nbytes // src_dtype.nbytes)])
        else:
            assert False

        return Tensor(new_shape, dst_dtype.name, self.device, self.storage)
    else:
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        dst_shape = [int(arg) for arg in args]
        return ops.reshape(self, dst_shape)


@register_method(torch.Tensor.contiguous)
def tensor_contiguous(self: Tensor) -> Tensor:
    # hidet tensor is always contiguous
    return self


@register_method(torch.Tensor.reshape)
def tensor_reshape(self: Tensor, *shape: int) -> Tensor:
    return ops.reshape(self, shape)


@register_method(torch.Tensor.split)
def tensor_split(self: Tensor, split_size, dim=0) -> List[Tensor]:
    parts: List[int] = []
    if isinstance(split_size, int):
        remain_size = self.shape[dim]
        while remain_size > 0:
            part_size = min(split_size, remain_size)
            parts.append(part_size)
            remain_size -= part_size
    else:
        assert isinstance(split_size, (list, tuple))
        parts = [int(v) for v in split_size]
        assert sum(parts) == self.shape[dim]
    return ops.split(self, axis=dim, parts=parts)


@register_method(torch.Tensor.squeeze)
def tensor_squeeze(self: Tensor, dim=None) -> Tensor:
    if dim is None:
        dims = [i for i, s in enumerate(self.shape) if s == 1]
        return ops.squeeze(self, dims)
    else:
        dim = int(dim)
        if self.shape[dim] != 1:
            return self
        else:
            return ops.squeeze(self, [dim])


@register_method(torch.Tensor.unsqueeze)
def tensor_unsqueeze(self: Tensor, dim) -> Tensor:
    return ops.unsqueeze(self, [int(dim)])


@register_method(torch.Tensor.type)
def tensor_type(self: Tensor, dtype: Union[str, torch.dtype], non_blocking: bool = False) -> Union[str, Tensor]:
    if dtype is None:
        return dtype_to_torch(self.dtype)
    else:
        _ = non_blocking
        return ops.cast(self, dtype_from_torch(dtype))


@register_method(torch.Tensor.expand)
def tensor_expand(self: Tensor, *sizes: int) -> Tensor:
    return ops.broadcast(self, sizes)


@register_method(torch.Tensor.masked_fill)
def tensor_masked_fill(self: Tensor, mask: Tensor, value: float) -> Tensor:
    return ops.where(mask, ops.full([], value, dtype=self.dtype, device=self.device), self)
