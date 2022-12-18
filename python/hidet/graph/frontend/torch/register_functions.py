from __future__ import annotations
from typing import Optional
import warnings
import operator
import torch
from hidet.graph.tensor import Tensor
from hidet.graph import ops
from hidet.utils import same_list
from .interpreter import register_function


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
def relu(x: Tensor):
    return ops.relu(x)


@register_function(torch.nn.functional.max_pool2d)
def max_pool2d(x: Tensor, kernel_size, stride, padding, dilation, ceil_mode, return_indices=False):
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
        warnings.warn(
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
