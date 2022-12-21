from __future__ import annotations
import torch
from hidet.ir.type import DataType
from hidet.graph.tensor import Tensor
from hidet.graph import ops
from .interpreter import register_method
from .utils import dtype_from_torch, device_from_torch


@register_method(torch.Tensor.cuda)
def tensor_cuda(self: Tensor):
    if self.is_symbolic():
        raise NotImplementedError('hidet: torch.Tensor.cuda() is not supported for symbolic tensors.')
    return self.cuda()


@register_method(torch.Tensor.cpu)
def tensor_cpu(self: Tensor):
    if self.is_symbolic():
        raise NotImplementedError('hidet: torch.Tensor.cpu() is not supported for symbolic tensors.')
    return self.cpu()


@register_method(torch.Tensor.to)
def tensor_to(self: Tensor, *args, **kwargs):
    """
    There are three argument format for torch.Tensor.to:

    1. to(self, dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
    2. to(self, device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
    3. to(self, non_blocking=False, copy=False, memory_format=torch.preserve_format)

    The current implementation should work for most use cases.

    See Also
        https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
    """
    if self.is_symbolic():
        raise NotImplementedError('hidet: Tensor.to() is not supported for symbolic tensors.')

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
def tensor_view(self: Tensor, *args):
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
            new_shape = self.shape[:-1] + [self.shape[-1] * (src_dtype.nbytes // dst_dtype.nbytes)]
        elif src_dtype.nbytes < dst_dtype.nbytes:
            assert dst_dtype.nbytes % src_dtype.nbytes == 0  # these should already have been checked by pytorch
            assert self.shape[-1] % (dst_dtype.nbytes // src_dtype.nbytes) == 0
            new_shape = self.shape[:-1] + [self.shape[-1] // (dst_dtype.nbytes // src_dtype.nbytes)]
        else:
            assert False

        return Tensor(new_shape, dst_dtype.name, self.device, self.storage)
    else:
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        dst_shape = [int(arg) for arg in args]
        return ops.reshape(self, dst_shape)


@register_method(torch.Tensor.contiguous)
def tensor_contiguous(self: Tensor):
    # hidet tensor is always contiguous
    return self
