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
from typing import Optional, Union, Sequence, Any, Tuple, List
import operator
import torch
from hidet.graph.tensor import Tensor, full_like, from_torch
from hidet.graph import ops
from hidet.utils import same_list
from hidet.ir.type import DataType
from hidet.ir.expr import Int
from hidet.runtime.device import Device
from .interpreter import register_function, register_method
from .interpreter import warnings
from .utils import dtype_from_torch, device_from_torch

Number = Union[int, float, bool]


@register_function(torch.nn.functional.conv1d)
def conv1d(x: Tensor, weight: Tensor, bias: Optional[Tensor], stride, padding, dilation, groups):
    x = ops.conv_pad(x, padding)
    y = ops.conv1d(x, weight, stride=stride, dilations=dilation, groups=groups)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2])
    return y


@register_function(torch.nn.functional.conv_transpose1d)
def conv1d_transpose(
    x: Tensor, weight: Tensor, bias: Optional[Tensor], stride, padding, output_padding, groups, dilation
):
    if dilation != 1 and not same_list(dilation, [1]):
        raise NotImplementedError("dilation != 1")
    y = ops.conv1d_transpose(x, weight, stride, padding, groups, output_padding)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2])
    return y


@register_function(torch.nn.functional.conv2d)
def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor], stride, padding, dilation, groups):
    x = ops.conv_pad(x, padding)
    y = ops.conv2d(x, weight, stride, dilation, groups)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2, 3])
    return y


@register_function(torch.nn.functional.conv_transpose2d)
def conv2d_transpose(
    x: Tensor, weight: Tensor, bias: Optional[Tensor], stride, padding, output_padding, groups, dilation
):
    if dilation != 1 and not same_list(dilation, [1, 1]):
        raise NotImplementedError("dilation != 1")
    y = ops.conv2d_transpose(x, weight, stride, padding, groups, output_padding)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2, 3])
    return y


@register_function(torch.nn.functional.conv3d)
def conv3d(x: Tensor, weight: Tensor, bias: Optional[Tensor], stride, padding, dilation, groups):
    x = ops.conv_pad(x, padding)
    y = ops.conv3d(x, weight, stride, dilation, groups)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2, 3, 4])
    return y


@register_function(torch.nn.functional.conv_transpose3d)
def conv3d_transpose(
    x: Tensor, weight: Tensor, bias: Optional[Tensor], stride, padding, output_padding, groups, dilation
):
    if dilation != 1 and not same_list(dilation, [1, 1, 1]):
        raise NotImplementedError("dilation != 1")
    y = ops.conv3d_transpose(x, weight, stride, padding, groups, output_padding)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2, 3, 4])
    return y


@register_function(torch.nn.functional.adaptive_avg_pool2d)
def adaptive_avg_pool2d(x: Tensor, output_size):
    return ops.adaptive_avg_pool2d(x, output_size)


@register_function(torch.nn.functional.relu)
def relu(x: Tensor, inplace: bool):
    # if inplace:
    #     warnings.warn_once('hidet: relu with inplace=True is not supported. Treat as inplace=False.')
    _ = inplace
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


@register_function(torch.nn.functional.max_pool3d)
def max_pool3d(x: Tensor, kernel_size, stride, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if dilation != 1 and not same_list(dilation, [1, 1, 1]):
        raise NotImplementedError("dilation != 1")
    if ceil_mode:
        raise NotImplementedError("ceil_mode=True")
    if return_indices:
        raise NotImplementedError("return_indices=True")
    y = ops.max_pool3d(x, kernel_size, stride, padding)
    return y


@register_function(torch.nn.functional.linear)
def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor]):
    y = ops.matmul(x, weight)
    if bias is not None:
        y = y + bias
    return y


@register_function(torch.nn.functional.bilinear)
def bilinear(x_1: Tensor, x_2: Tensor, weight: Tensor, bias: Optional[Tensor]):
    y = ops.matmul(x_1, ops.matmul(weight, x_2))
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
def cat(tensors: List[Tensor], dim: int):
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


@register_function(torch.nn.functional.avg_pool3d)
def avg_pool3d(x: Tensor, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if ceil_mode:
        raise NotImplementedError("ceil_mode=True")
    if not count_include_pad:
        raise NotImplementedError("count_include_pad=False")
    if divisor_override is not None:
        raise NotImplementedError("divisor_override is not None")
    y = ops.avg_pool3d(x, kernel_size, stride, padding)
    return y


@register_function(torch.nn.functional.interpolate)
def interpolate(
    input: Tensor,
    size: Union[int, Sequence[int]] = None,
    scale_factor=None,
    mode='nearest',
    align_corners=None,
    recompute_scale_factor=None,
    antialias=False,
):
    # please refer to the way that pytorch converts its interpolate function to onnx's resize operator
    # https://github.com/pytorch/pytorch/blob/940662c4dcaa090f20e39a63a8e319a58ca1460f/torch/onnx/symbolic_helper.py#L1133
    # for the details of how to convert pytorch's interpolate to hidet's resize operator as we are similar to onnx
    if len(input.shape) != 4:
        raise NotImplementedError("Currently only supports 4D inputs (NCHW)")

    if antialias:
        raise NotImplementedError("Currently does not support antialias=True")

    if (size is None) == (scale_factor is None):
        raise ValueError("Exactly one of size or scale_factor can be None")

    mode_hidet = mode
    if 'cubic' in mode:
        mode_hidet = 'cubic'
    if 'linear' in mode:
        mode_hidet = 'linear'

    if mode == 'nearest':
        coordinate_transformation_mode = 'asymmetric'
    elif align_corners:
        coordinate_transformation_mode = 'align_corners'
    else:
        coordinate_transformation_mode = 'half_pixel'

    return ops.resize2d(
        input,
        size=size,
        scale_factor=scale_factor,
        method=mode_hidet,
        coordinate_transformation_mode=coordinate_transformation_mode,
        rounding_method='floor',
        roi=None,
        cubic_alpha=-0.75,
        cubic_exclude=False,
        extrapolation_value=0.0,
        recompute_scale_factor=recompute_scale_factor,
    )


@register_function(operator.truediv)
def truediv(x: Union[Tensor, int, float], y: Union[Tensor, int, float]):
    import hidet

    def is_integer(v: Union[Tensor, int, float]) -> bool:
        return isinstance(v, int) or (isinstance(v, Tensor) and v.dtype.is_integer())

    if is_integer(x) and is_integer(y):
        if isinstance(y, (int, float)):
            y = hidet.asarray(y).to(device=x.device)
        return x / ops.cast(y, 'float32')
    else:
        return x / y


@register_function(operator.sub)
def sub(x: Tensor, y: Tensor):
    return x - y


@register_function(operator.neg)
def neg(x: Tensor):
    return -x


@register_function(torch.nn.functional.softmax)
@register_method(torch.Tensor.softmax)
def softmax(x: Tensor, dim: int, _stacklevel: int = 3, dtype=None):
    if dtype is not None:
        raise NotImplementedError("dtype is not None")
    return ops.softmax(x, dim)


@register_function(operator.matmul)
@register_function(torch.matmul)
def matmul(x: Tensor, y: Tensor):
    return ops.matmul(x, y)


@register_function(torch.zeros)
def zeros(*size, out=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
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

    _ = pin_memory
    _ = requires_grad

    return hidet.zeros(shape, dtype=dtype_from_torch(dtype), device=device_from_torch(device))


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
def gelu(x: Tensor, approximate: Optional[str] = "none"):
    if approximate is not None and approximate != "none":
        warnings.warn_once("hidet: gelu with approximate {repr(approximate)} is not supported. Treat as 'none'.")
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


@register_function(torch.nn.functional.group_norm)
def group_norm(
    x: Tensor,
    num_groups: int,
    num_channels: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    if x.shape[1] != num_channels:
        raise ValueError(
            "num_channels does not match tensor shape at index 2, expect {} but got {}".format(num_channels, x.shape[2])
        )
    if num_channels % num_groups != 0:
        raise ValueError("num_channels {} must be divisible by num_groups {}".format(num_channels, num_groups))

    y = ops.group_norm(x, num_groups, epsilon=eps)
    if weight is not None:
        y = y * weight.reshape([num_channels, 1, 1])
    if bias is not None:
        y = y + bias.reshape([num_channels, 1, 1])
    return y


@register_function(torch.tanh)
def tanh(x: Tensor):
    return ops.tanh(x)


@register_function(torch.nn.functional.hardtanh)
def hardtanh(x: Tensor, min_val: float, max_val: float):
    return ops.hardtanh(x, min_val, max_val)


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
@register_method(torch.Tensor.permute)
def permute(x: Tensor, *args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    dims = [int(v) for v in args]
    return ops.transpose(x, dims)


@register_function(torch.transpose)
@register_method(torch.Tensor.transpose)
def transpose(x: Tensor, dim0: int, dim1: int):
    if dim0 < dim1:
        dim0, dim1 = dim1, dim0
    return ops.transpose(x, [dim0, dim1])


@register_function(torch.nn.functional.dropout)
@register_function(torch.nn.functional.dropout1d)
@register_function(torch.nn.functional.dropout2d)
@register_function(torch.nn.functional.dropout3d)
def dropout(x: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False):
    _ = p
    if training:
        warnings.warn_once('hidet: Dropout/1D/2D/3D in training mode is not supported. Treating as in inference mode.')
    if inplace:
        warnings.warn_once("hidet: dropout(..., inplace=True) is not supported, treating as inplace=False")
    return x


@register_function(torch.nn.functional.relu6)
def relu6(x: Tensor, inplace: bool = False):
    if inplace:
        warnings.warn_once("hidet: relu6(..., inplace=True) is not supported, treating as inplace=False")
    return ops.relu6(x)


@register_function(torch.arange)
def arange(
    start: Number,
    end: Number,
    step: Number = 1,
    *,
    out: Optional[Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[Union[torch.device, str, None]] = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False,
):
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.arange(..., out=..., ...)")
    if layout is not None:
        raise NotImplementedError("hidet: does not support torch.arange(..., layout=..., ...)")
    if requires_grad and torch.is_grad_enabled():
        warnings.warn_once("hidet: requires_grad=True when torch.is_grad_enabled(), treating as requires_grad=False")
    _ = pin_memory  # ignore here, as hidet's default cpu memory is always pinned
    hidet_device: Device = device_from_torch(torch_device=device)
    hidet_dtype: DataType = dtype_from_torch(torch_dtype=dtype)
    return ops.arange(start, end, step, dtype=hidet_dtype, device=hidet_device)


@register_function(torch.addmm)
def addmm(
    input: Tensor, mat1: Tensor, mat2: Tensor, *, beta: Number = 1, alpha: Number = 1, out: Optional[Tensor] = None
):
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.addmm(..., out=..., ...)")
    y = ops.matmul(mat1, mat2)
    if alpha not in [1, 1.0]:
        y = y * alpha
    if beta not in [1, 1.0]:
        input = input * beta
    return y + input


@register_function(torch.where)
def where(condition: Tensor, x: Tensor, y: Tensor):
    return ops.where(cond=condition, x=x, y=y)


@register_function(torch.pow)
def pow(base: Tensor, exponent: Union[Number, Tensor]):
    if isinstance(exponent, (int, float, bool)):
        exponent = full_like(base, exponent)
    return ops.pow(base, exponent)


@register_function(torch.full)
def full(size, fill_value, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.full(..., out=..., ...)")
    if layout not in [None, torch.strided]:
        raise NotImplementedError("hidet: does not support torch.full(..., layout=..., ...)")
    if requires_grad and torch.is_grad_enabled():
        warnings.warn_once("hidet: requires_grad=True when torch.is_grad_enabled(), treating as requires_grad=False")
    hidet_device: Device = device_from_torch(torch_device=device)
    hidet_dtype: DataType = dtype_from_torch(torch_dtype=dtype)
    return ops.full(size, fill_value, dtype=hidet_dtype, device=hidet_device)


@register_function(torch.empty)
def empty(
    *size,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
):
    import hidet

    if out is not None:
        raise NotImplementedError("hidet: does not support torch.empty(..., out=..., ...)")
    if layout not in [None, torch.strided]:
        raise NotImplementedError("hidet: does not support torch.empty(..., layout=..., ...)")
    if requires_grad and torch.is_grad_enabled():
        warnings.warn_once("hidet: requires_grad=True when torch.is_grad_enabled(), treating as requires_grad=False")
    if pin_memory:
        raise NotImplementedError("hidet: does not support torch.empty(..., pin_memory=True, ...)")
    if memory_format != torch.contiguous_format:
        raise NotImplementedError("hidet: does not support torch.empty(..., memory_format=..., ...)")

    hidet_device: Device = device_from_torch(torch_device=device)
    hidet_dtype: DataType = dtype_from_torch(torch_dtype=dtype)
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    return hidet.empty(size, dtype=hidet_dtype, device=hidet_device)


@register_function(torch.bmm)
def bmm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.bmm(..., out=...)")
    return ops.matmul(input, mat2)


@register_function(torch.baddbmm)
def baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out: Optional[Tensor] = None) -> Tensor:
    import hidet

    if out is not None:
        raise NotImplementedError("hidet: does not support torch.baddbmm(..., out=...)")

    if alpha == 0 and beta == 0:
        size = batch1.shape[0:2] + [batch2.shape[-1]]
        return hidet.zeros(shape=size, dtype=input.dtype, device=input.device)
    elif alpha == 0:
        return beta * input
    elif beta == 0:
        return alpha * ops.matmul(batch1, batch2)

    if alpha == 1 and beta == 1:
        return input + ops.matmul(batch1, batch2)
    elif alpha == 1:
        return beta * input + ops.matmul(batch1, batch2)
    elif beta == 1:
        return input + alpha * ops.matmul(batch1, batch2)

    return beta * input + alpha * ops.matmul(batch1, batch2)


@register_function(torch.tensor)
def torch_tensor(
    data: Any, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None, requires_grad: bool = False
) -> Tensor:
    if requires_grad and torch.is_grad_enabled():
        warnings.warn_once("hidet: requires_grad=True when torch.is_grad_enabled(), treating as requires_grad=False")
    if isinstance(data, Tensor):
        device = device_from_torch(torch_device=device) if device is not None else device
        return data.to(device=device, dtype=dtype_from_torch(dtype))
    else:
        tt = torch.tensor(data, dtype=dtype, device=device)
        return from_torch(tt)


@register_function(torch.sigmoid)
def sigmoid(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        warnings.warn_once("hidet: does not support torch.sigmoid(..., out=...)")
    return ops.sigmoid(x)


@register_function(torch.exp)
def exp(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        warnings.warn_once("hidet: does not support torch.exp(..., out=...)")
    return ops.exp(x)


@register_function(torch.nn.functional.hardsigmoid)
def hardsigmoid(x: Tensor, inplace: bool):
    if inplace:
        warnings.warn_once('hidet: hardsigmoid with inplace=True is not supported. Treat as inplace=False.')
    return ops.hardsigmoid(x)


@register_function(torch.nn.functional.silu)
def silu(x: Tensor, inplace: bool):
    if inplace:
        warnings.warn_once('hidet: silu with inplace=True is not supported. Treat as inplace=False.')
    return ops.silu(x)


@register_function(torch.nn.functional.hardswish)
def hardswish(x: Tensor, inplace: bool):
    if inplace:
        warnings.warn_once('hidet: hardswish with inplace=True is not supported. Treat as inplace=False.')
    return ops.hardswish(x)


@register_function(torch.nn.functional.softmin)
def softmin(x: Tensor, axis: int):
    return ops.softmin(x, axis)


@register_function(torch.nn.functional.softplus)
def softplus(x: Tensor, beta: int, threshold: int):
    return ops.softplus(x, beta, threshold)


@register_function(torch.nn.functional.softshrink)
def softshrink(x: Tensor, lambda_val: float):
    return ops.softshrink(x, lambda_val)


@register_function(torch.nn.functional.tanhshrink)
def tanhshrink(x: Tensor):
    return ops.tanhshrink(x)


@register_function(torch.nn.functional.hardshrink)
def hardshrink(x: Tensor, lambda_val: float):
    return ops.hardshrink(x, lambda_val)


@register_function(torch.nn.functional.softsign)
def softsign(x: Tensor):
    return ops.softsign(x)


@register_function(torch.nn.functional.celu)
def celu(x: Tensor, alpha: float):
    return ops.celu(x, alpha)


@register_function(torch.nn.functional.logsigmoid)
def logsigmoid(x: Tensor):
    return ops.logsigmoid(x)


@register_function(torch.nn.functional.mish)
def mish(x: Tensor, inplace: bool = False):
    if inplace:
        warnings.warn_once('hidet: mish with inplace=True is not supported. Treat as inplace=False.')
    return ops.multiply(x, ops.tanh(ops.softplus(x, 1, 20)))


@register_function(torch.gather)
def gather(x: Tensor, dim: int, index: Tensor, *, sparse_grad=False, out=None):
    if sparse_grad:
        warnings.warn_once('hidet: gather with sparse_grad=True is not supported. Treat as sparse_grad=False.')
    if out is not None:
        raise NotImplementedError('hidet: gather with out=... is not supported')
    return ops.gather(x, index, axis=dim)


@register_function(torch.maximum)
def maximum(x: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.maximum(..., out=...)")
    return ops.maximum(x, other)


@register_function(torch.minimum)
def minimum(x: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.minimum(..., out=...)")
    return ops.minimum(x, other)


@register_function(torch.max)
def torch_max_v1(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.max(..., out=...)")
    return ops.max(x, dims=list(range(len(x.shape))), keep_dim=True)


@register_function(torch.max)
def torch_max_v2(
    x: Tensor, other: Union[Tensor, int], *, out: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.max(..., out=...)")
    if isinstance(other, Tensor):
        return ops.maximum(x, other)
    else:
        return torch_max_v3(x, other)


@register_function(torch.max)
def torch_max_v3(
    x: Tensor, dim: Int, keepdim: bool = False, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor]] = None
) -> Tuple[Tensor, Tensor]:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.max(..., out=...)")
    values = ops.max(x, dims=dim, keep_dim=keepdim)
    indices = ops.argmax(x, dim=dim, keep_dim=keepdim)
    return values, indices


@register_function(torch.min)
def torch_min_v1(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.min(..., out=...)")
    return ops.min(x, dims=list(range(len(x.shape))), keep_dim=True)


@register_function(torch.min)
def torch_min_v2(
    x: Tensor, other: Union[Tensor, int], *, out: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.min(..., out=...)")
    if isinstance(other, Tensor):
        return ops.minimum(x, other)
    else:
        return torch_min_v3(x, other)


@register_function(torch.min)
def torch_min_v3(
    x: Tensor, dim: Int, keepdim: bool = False, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor]] = None
) -> Tuple[Tensor, Tensor]:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.min(..., out=...)")
    values = ops.min(x, dims=dim, keep_dim=keepdim)
    indices = ops.argmin(x, dim=dim, keep_dim=keepdim)
    return values, indices
