from __future__ import annotations
from typing import Optional, Union, Sequence
import operator
import torch
from hidet.graph.tensor import Tensor
from hidet.graph import ops
from hidet.utils import same_list
from hidet.ir.type import data_type
from .interpreter import register_function, register_method
from .interpreter import warnings
from .utils import dtype_from_torch, device_from_torch


@register_function(torch.nn.functional.conv2d)
def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor], stride, padding, dilation, groups):
    if dilation != 1 and not same_list(dilation, [1, 1]):
        raise NotImplementedError("dilation != 1")
    x = ops.conv_pad(x, padding)
    y = ops.conv2d(x, weight, stride, groups)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2, 3])
    return y


@register_function(torch.nn.functional.adaptive_avg_pool2d)
def adaptive_avg_pool2d(x: Tensor, output_size):
    return ops.adaptive_avg_pool2d(x, output_size)


@register_function(torch.nn.functional.relu)
def relu(x: Tensor, inplace: bool):
    if inplace:
        warnings.warn_once('hidet: relu with inplace=True is not supported. Treat as inplace=False.')
    return ops.relu(x)


@register_function(torch.nn.functional.max_pool2d)
def max_pool2d(x: Tensor, kernel_size, stride, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if dilation != 1 and not same_list(dilation, [1, 1]):
        raise NotImplementedError("dilation != 1")
    if ceil_mode:
        raise NotImplementedError("ceil_mode=True")
    if return_indices:
        raise NotImplementedError("return_indices=True")
    y = ops.max_pool2d(x, kernel_size, stride, padding)
    return y


@register_function(torch.nn.functional.linear)
def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor]):
    y = ops.matmul(x, weight)
    if bias is not None:
        y = y + bias
    return y


@register_function(operator.add)
def add(x: Tensor, y: Tensor):
    return ops.add(x, y)


@register_function(operator.iadd)
def iadd(x: Tensor, y: Tensor):
    return ops.add(x, y)


@register_function(torch.sin)
def sin(x: Tensor):
    return ops.sin(x)


@register_function(torch.cos)
def cos(x: Tensor):
    return ops.cos(x)


@register_function(torch.nn.functional.batch_norm)
def batch_norm(
    x: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
):
    if training:
        warnings.warn_once(
            "hidet: batch_norm training=True is not supported, treat it as training=False. "
            "Results may be different from PyTorch. "
            "Please set the module to eval mode via .eval() first to silence this warning."
        )
    y = ops.batch_norm_infer(x, running_mean, running_var, epsilon=eps)
    _ = momentum  # unused
    if weight is not None:
        y = y * weight.unsqueeze([0, 2, 3])
    if bias is not None:
        y = y + bias.unsqueeze([0, 2, 3])
    return y


@register_function(torch.flatten)
def flatten(x: Tensor, start_dim: int, end_dim: int = -1):
    return ops.flatten(x, start_dim, end_dim)


@register_function(operator.getitem)
def getitem(x: Tensor, index):
    return x[index]


@register_function(operator.mul)
def mul(x: Tensor, y: Tensor):
    return x * y


@register_function(torch.cat)
def cat(tensors: list[Tensor], dim: int):
    return ops.concat(tensors, dim)


@register_function(torch.unsqueeze)
def unsqueeze(x: Tensor, dim: int):
    return ops.unsqueeze(x, [dim])


@register_function(torch.nn.functional.avg_pool2d)
def avg_pool2d(x: Tensor, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if ceil_mode:
        raise NotImplementedError("ceil_mode=True")
    if not count_include_pad:
        raise NotImplementedError("count_include_pad=False")
    if divisor_override is not None:
        raise NotImplementedError("divisor_override is not None")
    y = ops.avg_pool2d(x, kernel_size, stride, padding)
    return y


@register_function(operator.truediv)
def truediv(x: Union[Tensor, int, float], y: Union[Tensor, int, float]):
    import hidet

    def is_integer(v: Union[Tensor, int, float]) -> bool:
        return isinstance(v, int) or (isinstance(v, Tensor) and data_type(v.dtype).is_integer())

    if is_integer(x) and is_integer(y):
        if isinstance(y, (int, float)):
            y = hidet.array(y).to(device=x.device)
        return x / ops.cast(y, 'float32')
    else:
        return x / y


@register_function(operator.sub)
def sub(x: Tensor, y: Tensor):
    return x - y


@register_function(torch.nn.functional.softmax)
def softmax(x: Tensor, dim: int, dtype=None):
    if dtype is not None:
        raise NotImplementedError("dtype is not None")
    return ops.softmax(x, dim)


@register_function(torch.matmul)
def matmul(x: Tensor, y: Tensor):
    return ops.matmul(x, y)


@register_function(torch.ones)
def ones(
    *size: Union[int, Sequence[int]],
    out: Optional[Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[Union[torch.device, str, None]] = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False,
):
    import hidet

    if out is not None:
        raise NotImplementedError("out is not None")
    if layout is not None:
        raise NotImplementedError("layout is not None")

    if len(size) == 1:
        if isinstance(size[0], (list, tuple)):
            size = size[0]

    shape = [int(v) for v in size]
    if dtype is None:
        dtype = torch.get_default_dtype()

    # currently, hidet's default cpu memory is always pinned.
    # todo: fix here when hidet supports non-pinned memory
    _ = pin_memory
    _ = requires_grad

    return hidet.ones(
        shape=shape, dtype=dtype_from_torch(torch_dtype=dtype).name, device=device_from_torch(torch_device=device)
    )


@register_function(torch.nn.functional.gelu)
def gelu(x: Tensor):
    return ops.gelu(x)


@register_function(torch.nn.functional.layer_norm)
def layer_norm(
    x: Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    y = ops.layer_norm(x, num_last_dims=len(normalized_shape), epsilon=eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


@register_function(torch.tanh)
def tanh(x: Tensor):
    return ops.tanh(x)


@register_function(torch.nn.functional.embedding)
def embedding(
    x: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):
    import hidet

    assert len(weight.shape) == 2
    if max_norm is not None:
        # normalize the whole embedding matrix, as we are doing inference
        p = hidet.full([], norm_type, dtype=weight.dtype, device=weight.device)
        invp = hidet.full([], 1.0 / norm_type, dtype=weight.dtype, device=weight.device)
        eps = hidet.full([], 1e-8, dtype=weight.dtype, device=weight.device)
        d = ops.pow(weight, p).sum(-1)
        d = ops.pow(d, invp).unsqueeze(-1) + eps
        weight = weight / d

    _ = padding_idx  # unused
    _ = scale_grad_by_freq  # unused
    _ = sparse  # unused

    y = ops.take(weight, x, axis=0)
    return y


@register_function(torch.permute)
@register_method(Tensor, 'permute')
def permute(x: Tensor, *args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    dims = [int(v) for v in args]
    return ops.transpose(x, dims)


@register_function(torch.transpose)
@register_method(Tensor, 'transpose')
def transpose(x: Tensor, dim0: int, dim1: int):
    if dim0 < dim1:
        dim0, dim1 = dim1, dim0
    return ops.transpose(x, [dim0, dim1])
