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

from hidet.ir.type import DataType, Int
from hidet.ir.expr import Expr
from hidet.graph.tensor import Tensor
from hidet.graph import ops
from hidet.runtime.device import instantiate_device
from .registry import register_method
from .utils import dtype_from_torch, device_from_torch, dtype_to_torch


@register_method(torch.Tensor.cuda)
def tensor_cuda(self: Tensor) -> Tensor:
    return self.cuda()


@register_method(torch.Tensor.cpu)
def tensor_cpu(self: Tensor) -> Tensor:
    return self.cpu()


@register_method(torch.Tensor.int)
def tensor_int(self: Tensor) -> Tensor:
    return ops.cast(self, "int32")


@register_method(torch.Tensor.long)
def tensor_long(self: Tensor) -> Tensor:
    return ops.cast(self, "int64")


@register_method(torch.Tensor.float)
def tensor_float(self: Tensor) -> Tensor:
    return ops.cast(self, "float32")


@register_method(torch.Tensor.half)
def tensor_half(self: Tensor) -> Tensor:
    return ops.cast(self, "float16")


@register_method(torch.Tensor.bfloat16)
def tensor_bfloat16(self: Tensor) -> Tensor:
    return ops.cast(self, "bfloat16")


@register_method(torch.Tensor.bool)
def tensor_bool(self: Tensor) -> Tensor:
    return ops.cast(self, "bool")


@register_method(torch.Tensor.type_as)
def tensor_type_as(self: Tensor, other: Tensor) -> Tensor:
    return ops.cast(self, other.dtype)


@register_method(torch.Tensor.fill_)
def fill_(self: Tensor, value):
    return ops.full(self.shape, value, dtype=self.dtype, device=self.device)


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
            device = arg
        elif isinstance(arg, Tensor):
            dtype = arg.dtype
            if self.is_symbolic() and arg.device != self.device:
                raise NotImplementedError('hidet: Tensor.to(..., device=...) is not supported for symbolic tensors.')
            device = arg.device
        else:
            raise ValueError(f'Unsupported argument type: {type(arg)}')

    if memory_format != torch.preserve_format:
        raise NotImplementedError('hidet: torch.Tensor.to(..., memory_format=..., ...) is not supported.')

    _ = copy
    _ = non_blocking

    temp = self.to(dtype=dtype_from_torch(dtype).name if dtype else None)
    if self.is_symbolic() and device is not None and instantiate_device(device_from_torch(device)) != self.device:
        return ops.transfer(temp, dst_device=device_from_torch(device))
    return temp.to(device=device_from_torch(device) if device is not None else None)


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
        dst_shape = []
        for v in args:
            if isinstance(v, torch.Tensor):
                v = v.item()
            elif isinstance(v, Expr):
                # do nothing
                pass
            else:
                v = int(v)
            dst_shape.append(v)
        return ops.reshape(self, dst_shape)


@register_method(torch.Tensor.view_as)
def torch_view_as(self: Tensor, other: Tensor) -> Tensor:
    return ops.reshape(self, other.shape)


@register_method(torch.Tensor.contiguous)
def tensor_contiguous(self: Tensor) -> Tensor:
    # hidet tensor is always contiguous
    return self


@register_method(torch.Tensor.reshape)
def tensor_reshape(self: Tensor, *shape: int) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    return ops.reshape(self, shape)


@register_method(torch.Tensor.size)
def tensor_size(self: Tensor, dim=None) -> List[Int]:
    if dim is None:
        return self.shape
    else:
        return self.shape[dim]


@register_method(torch.Tensor.type)
def tensor_type(self: Tensor, dtype: Union[str, torch.dtype], non_blocking: bool = False) -> Union[str, Tensor]:
    if dtype is None:
        return dtype_to_torch(self.dtype)
    else:
        _ = non_blocking
        return ops.cast(self, dtype_from_torch(dtype))


@register_method(torch.Tensor.expand)
def tensor_expand(self: Tensor, *sizes: int) -> Tensor:
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        sizes = list(sizes[0])
    else:
        sizes: List[int] = list(sizes)
    assert len(sizes) >= len(self.shape)
    for i in range(len(sizes)):
        if sizes[i] == -1:
            ri = len(sizes) - 1 - i
            assert ri < len(self.shape)
            sizes[i] = self.shape[len(self.shape) - 1 - ri]
    return ops.broadcast(self, sizes)


@register_method(torch.Tensor.expand_as)
def tensor_expand_as(self: Tensor, other: Tensor) -> Tensor:
    return ops.broadcast(self, other.shape)


@register_method(torch.Tensor.masked_fill)
def tensor_masked_fill(self: Tensor, mask: Tensor, value: float) -> Tensor:
    return ops.where(mask, ops.full([], value, dtype=self.dtype, device=self.device), self)


@register_method(torch.Tensor.masked_fill_)
def tensor_masked_fill_(self: Tensor, mask: Tensor, value: float) -> Tensor:
    return ops.where(mask, ops.full([], value, dtype=self.dtype, device=self.device), self)


@register_method(torch.Tensor.repeat)
def tensor_repeat(self: Tensor, *sizes: int) -> Tensor:
    if len(self.shape) < len(sizes):
        shape = [1] * (len(sizes) - len(self.shape)) + list(self.shape)
        x = ops.reshape(self, shape)
        return ops.tile(x, sizes)

    return ops.tile(self, sizes)


@register_method(torch.Tensor.detach)
def tensor_detach(self: Tensor) -> Tensor:
    return self


# Turns out torch.Tensor.all/any is slightly different from torch.all
# because of the default values of dim and keepdim
@register_method(torch.Tensor.all)
def tensor_all(self: Tensor, dim=None, keepdim=False) -> Tensor:
    return ops.all(self, axis=dim, keepdims=keepdim)


@register_method(torch.Tensor.any)
def tensor_any(self: Tensor, dim=None, keepdim=False) -> Tensor:
    return ops.any(self, axis=dim, keepdims=keepdim)


@register_method(torch.Tensor.new_zeros)
def tensor_new_zeros(self: Tensor, *size, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
    if layout is not None:
        raise NotImplementedError("layout is not None")
    if len(size) == 1:
        if isinstance(size[0], (list, tuple)):
            size = size[0]
    shape = size
    if dtype is None:
        dtype = self.dtype
    if device is None:
        device = self.device

    _ = pin_memory
    _ = requires_grad

    return ops.full(shape, dtype=dtype, device=device, value=dtype.zero)


@register_method(torch.Tensor.new_full)
def tensor_new_full(
    self: Tensor, size, fill_value, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False
):
    if layout is not None:
        raise NotImplementedError("layout is not None")
    if dtype is None:
        dtype = self.dtype
    if device is None:
        device = self.device

    _ = pin_memory
    _ = requires_grad

    return ops.full(size, dtype=dtype, device=device, value=fill_value)


@register_method(torch.Tensor.zero_)
def tensor_zero_(self: Tensor):
    return ops.full(self.shape, dtype=self.dtype, device=self.device, value=self.dtype.zero)


@register_method(torch.Tensor.new_ones)
def tensor_new_ones(self: Tensor, size, *, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False):
    if layout is not None and layout != torch.strided:
        raise NotImplementedError("layout is not None and layout != torch.strided")
    if len(size) == 1:
        if isinstance(size[0], (list, tuple)):
            size = size[0]
    shape = size
    if dtype is None:
        dtype = self.dtype
        device = self.device
    _ = pin_memory
    _ = requires_grad

    return ops.full(shape, dtype=dtype, device=device, value=dtype.one)


@register_method(torch.Tensor.new_ones)
def tensor_new_ones_v2(
    self: Tensor, *size, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False
):
    if layout is not None and layout != torch.strided:
        raise NotImplementedError("layout is not None and layout != torch.strided")
    if len(size) == 1:
        if isinstance(size[0], (list, tuple)):
            size = size[0]
    shape = size
    if dtype is None:
        dtype = self.dtype
        device = self.device
    _ = pin_memory
    _ = requires_grad

    return ops.full(shape, dtype=dtype, device=device, value=dtype.one)
