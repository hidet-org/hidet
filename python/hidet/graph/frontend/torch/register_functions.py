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
# pylint: disable=protected-access, c-extension-no-member, function-redefined
from typing import Optional, Union, Sequence, Any, Tuple, List
import operator
import functools
import torch
from hidet.graph.tensor import Tensor, full_like, from_torch
from hidet.graph import ops
from hidet.utils import same_list
from hidet.ir.type import DataType
from hidet.ir import expr
from hidet.ir.dtypes import promote_type
from hidet.ir.expr import Expr, Int, is_constant
from hidet.runtime.device import Device
from .interpreter import register_function, register_method
from .interpreter import warnings
from .utils import dtype_from_torch, device_from_torch, normalize_to_scalar, convert_to_scalar_if_possible

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
    y = ops.conv2d(x, weight, stride, dilation, groups, padding=padding)
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


@register_function(torch.nn.functional.adaptive_avg_pool3d)
def adaptive_avg_pool3d(x: Tensor, output_size):
    return ops.adaptive_avg_pool3d(x, output_size)


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
    if return_indices:
        raise NotImplementedError("return_indices=True")
    y = ops.max_pool2d(x, kernel_size, stride, padding, ceil_mode=ceil_mode)
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
def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor], weight_is_transposed=False):
    if len(weight.shape) > 1 and not weight_is_transposed:
        weight = ops.transpose(weight, [1, 0])
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
@register_function(torch.ops.aten.add.Tensor)
def add(x: Tensor, y: Tensor):
    return x + y


@register_function(operator.iadd)
def iadd(x: Tensor, y: Tensor):
    return x + y


@register_function(torch.sin)
@register_function(torch.ops.aten.sin.default)
def sin(x: Tensor):
    return ops.sin(x)


@register_function(torch.cos)
@register_function(torch.ops.aten.cos.default)
def cos(x: Tensor):
    return ops.cos(x)


@register_function(operator.not_)
def not_(x: Union[Tensor, Expr]):
    if isinstance(x, Tensor):
        return ops.logical_not(x)
    elif isinstance(x, Expr):
        return expr.logical_not(x)
    else:
        return not x


@register_function(operator.and_)
def and_(x: Union[Tensor, Expr], y: Union[Tensor, Expr]):
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return ops.logical_and(x, y)
    else:
        return expr.logical_and(x, y)


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
    if len(x.shape) == 3:
        dims = [0, 2]
    if len(x.shape) == 4:
        dims = [0, 2, 3]
    elif len(x.shape) == 5:
        dims = [0, 2, 3, 4]
    else:
        raise NotImplementedError("batch_norm only accepts 3D, 4D, 5D input")
    if weight is not None:
        y = y * weight.unsqueeze(dims)
    if bias is not None:
        y = y + bias.unsqueeze(dims)
    return y


@register_function(torch.flatten)
def flatten(x: Tensor, start_dim: int, end_dim: int = -1):
    return ops.flatten(x, start_dim, end_dim)


@register_function(operator.getitem)
def getitem(x: Tensor, index):
    return x[index]


@register_function(operator.setitem)
def setitem(x: Tensor, item, setvalue):

    if isinstance(item, list):
        item = tuple(item)
    if not isinstance(item, tuple):
        item = tuple([item])

    if not isinstance(setvalue, (int, float)):
        raise NotImplementedError('Currently Tensor __setitem__ only supports int or float values')

    # now, the item could have
    # 1. integer index
    # 2. slice
    # 3. Ellipsis
    # 4. None
    # e.g., [1, 3:5, ..., None]

    # process Ellipsis
    # e.g., x[1, ..., 2] -> x[1, :, :, 2]
    if Ellipsis in item:
        if item.count(Ellipsis) > 1:
            raise ValueError('Only one ellipsis allowed in index.')
        ellipsis_index = item.index(Ellipsis)
        ellipsis_ndim = len(x.shape) - sum([1 if axis not in [None, Ellipsis] else 0 for axis in item])
        ellipsis_ndim = max(ellipsis_ndim, 0)
        item = item[:ellipsis_index] + (slice(None),) * ellipsis_ndim + item[ellipsis_index + 1 :]

    # normalize index
    normalized_item = []
    for i, v in enumerate(item):
        if isinstance(v, int):
            if v < 0:
                v = v + x.shape[i]
            if is_constant(v, x.shape[i]) and (v < 0 or v >= x.shape[i]):
                raise IndexError('index {} is out of bound for dimension {} with size {}'.format(v, i, x.shape[i]))
            normalized_item.append(v)
        elif v is not None:
            # None affects getitem, but is ignored in setitem
            normalized_item.append(v)
    item = tuple(normalized_item)

    # process slice and integer index
    rank = len(x.shape)
    while len(item) < rank:
        item = item + (slice(None),)
    starts, ends, steps = [], [], []
    squeeze_dims = []
    for dim, v in enumerate(item):
        if isinstance(v, (int, Expr)):
            squeeze_dims.append(dim)
            starts.append(v)
            ends.append(v + 1)
            steps.append(1)
        else:
            assert isinstance(v, slice)
            starts.append(v.start)
            ends.append(v.stop)
            steps.append(v.step)

    out = ops.set_strided_slice(x, starts, ends, steps, setvalue)
    return out


@register_function(operator.mul)
@register_function(torch.mul)
@register_function(torch.ops.aten.mul.Tensor)
def mul(x: Tensor, y: Tensor):
    return x * y


@register_function(torch.cat)
def cat(tensors: List[Tensor], dim: int):
    dtype = functools.reduce(promote_type, [t.dtype for t in tensors])
    tensors = [ops.cast(t, dtype) for t in tensors]
    return ops.concat(tensors, dim)


@register_function(torch.cat)
def cat(tensors: List[Tensor], axis: int):  # PyTorch supports axis as well as the argument name
    dtype = functools.reduce(promote_type, [t.dtype for t in tensors])
    tensors = [ops.cast(t, dtype) for t in tensors]
    return ops.concat(tensors, axis)


@register_function(torch.unsqueeze)
def unsqueeze(x: Tensor, dim: int):
    return ops.unsqueeze(x, [dim])


@register_function(torch.nn.functional.avg_pool2d)
def avg_pool2d(
    x: Tensor, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None
):
    if ceil_mode:
        raise NotImplementedError("ceil_mode=True")
    if not count_include_pad:
        raise NotImplementedError("count_include_pad=False")
    if divisor_override is not None:
        raise NotImplementedError("divisor_override is not None")
    if stride is None:
        stride = kernel_size
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
@register_function(torch.neg)
@register_function(torch.ops.aten.neg.default)
def neg(x: Tensor):
    return -x


@register_function(torch.nn.functional.softmax)
@register_method(torch.Tensor.softmax)
def softmax(x: Tensor, dim: int, _stacklevel: int = 3, dtype=None):
    if dtype is not None:
        x = ops.cast(x, dtype_from_torch(dtype))
    return ops.softmax(x, dim)


@register_function(operator.matmul)
@register_function(torch.matmul)
def matmul(x: Tensor, y: Tensor):
    return ops.matmul(x, y)


@register_function(torch.zeros)
def zeros(*size, out=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
    if out is not None:
        raise NotImplementedError("out is not None")
    if layout is not None:
        raise NotImplementedError("layout is not None")
    if len(size) == 1:
        if isinstance(size[0], (list, tuple)):
            size = size[0]
    shape = size
    if dtype is None:
        dtype = torch.get_default_dtype()

    _ = pin_memory
    _ = requires_grad

    device = device_from_torch(device)
    dtype = dtype_from_torch(dtype)

    return ops.full(shape, dtype=dtype, device=device, value=dtype.zero)


@register_function(torch.ones)
def ones(
    *size: Union[Int, Sequence[Int]],
    out: Optional[Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[Union[torch.device, str, None]] = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False,
):
    if out is not None:
        raise NotImplementedError("out is not None")
    if layout is not None:
        raise NotImplementedError("layout is not None")

    if len(size) == 1:
        if isinstance(size[0], (list, tuple)):
            size = size[0]

    shape = [v if isinstance(v, Expr) else int(v) for v in size]
    if dtype is None:
        dtype = torch.get_default_dtype()

    _ = pin_memory
    _ = requires_grad

    dtype = dtype_from_torch(dtype)
    device = device_from_torch(device)

    return ops.full(shape=shape, dtype=dtype, device=device, value=dtype.one)


@register_function(torch.nn.functional.gelu)
def gelu(x: Tensor, approximate: Optional[str] = "none"):
    approximate = {"none": False, "tanh": True}[approximate]
    return ops.gelu(x, approximate=approximate)


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


@register_function(torch.swapaxes)
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
    end: Number = None,
    step: Number = 1,
    *,
    out: Optional[Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[Union[torch.device, str, None]] = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False,
):
    if end is None:
        end = start
        start = 0
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.arange(..., out=..., ...)")
    if layout is not None:
        raise NotImplementedError("hidet: does not support torch.arange(..., layout=..., ...)")
    if requires_grad and torch.is_grad_enabled():
        warnings.warn_once("hidet: requires_grad=True when torch.is_grad_enabled(), treating as requires_grad=False")
    _ = pin_memory  # ignore here, as hidet's default cpu memory is always pinned
    hidet_device: Device = device_from_torch(torch_device=device)
    hidet_dtype: DataType = dtype_from_torch(torch_dtype=dtype)
    start = normalize_to_scalar(start)
    end = normalize_to_scalar(end)
    step = normalize_to_scalar(step)
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
def where(condition: Tensor, x: Union[Tensor, Number], y: Union[Tensor, Number]):
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
    return ops.full(size, dtype=hidet_dtype, device=hidet_device, value=hidet_dtype.zero)


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
@register_method(torch.Tensor.exp)
def exp(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        warnings.warn_once("hidet: does not support torch.exp(..., out=...)")
    return ops.exp(x)


@register_function(torch.nn.functional.hardsigmoid)
def hardsigmoid(x: Tensor, inplace: bool = False):
    if inplace:
        warnings.warn_once('hidet: hardsigmoid with inplace=True is not supported. Treat as inplace=False.')
    return ops.hardsigmoid(x)


@register_function(torch.nn.functional.silu)
def silu(x: Tensor, inplace: bool = False):
    if inplace:
        warnings.warn_once('hidet: silu with inplace=True is not supported. Treat as inplace=False.')
    return ops.silu(x)


@register_function(torch.nn.functional.hardswish)
def hardswish(x: Tensor, inplace: bool = False):
    if inplace:
        warnings.warn_once('hidet: hardswish with inplace=True is not supported. Treat as inplace=False.')
    return ops.hardswish(x)


@register_function(torch.nn.functional.softmin)
def softmin(x: Tensor, dim: int):
    return ops.softmin(x, dim)


@register_function(torch.nn.functional.softplus)
def softplus(x: Tensor, beta: int = 1, threshold: int = 20):
    return ops.softplus(x, beta, threshold)


@register_function(torch.nn.functional.softshrink)
def softshrink(x: Tensor, lambd=0.5):
    return ops.softshrink(x, lambd)


@register_function(torch.nn.functional.tanhshrink)
def tanhshrink(x: Tensor):
    return ops.tanhshrink(x)


@register_function(torch.nn.functional.hardshrink)
def hardshrink(x: Tensor, lambd=0.5):
    return ops.hardshrink(x, lambd)


@register_function(torch.nn.functional.softsign)
def softsign(x: Tensor):
    return ops.softsign(x)


@register_function(torch.nn.functional.celu)
def celu(x: Tensor, alpha: float = 1.0, inplace: bool = False):
    if inplace:
        warnings.warn_once('hidet: celu with inplace=True is not supported. Treat as inplace=False.')
    return ops.celu(x, alpha)


@register_function(torch.nn.functional.logsigmoid)
def logsigmoid(x: Tensor):
    return ops.logsigmoid(x)


@register_function(torch.nn.functional.mish)
def mish(x: Tensor, inplace: bool = False):
    if inplace:
        warnings.warn_once('hidet: mish with inplace=True is not supported. Treat as inplace=False.')
    return ops.multiply(x, ops.tanh(ops.softplus(x, 1, 20)))


@register_function(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor = None, dropout_p: float = 0.0, is_causal: bool = False
):
    import math

    if not math.isclose(dropout_p, 0.0):
        warnings.warn_once('hidet: attention dropout is not supported. Treat as dropout_p=0.0')

    k_rank = len(k.shape)
    # transpose last 2 dimensions of k, and normalize by sqrt(head_dim)
    k_transpose_scaled = ops.transpose(k, [i for i in range(k_rank - 2)] + [k_rank - 1, k_rank - 2]) / math.sqrt(
        k.shape[-1]
    )

    from hidet import boolean, float16

    type_match = (
        q.dtype == k.dtype == v.dtype == float16
        and len(q.shape) == len(k_transpose_scaled.shape) == len(v.shape)
        and k_transpose_scaled.shape[-1] == v.shape[-2]
        and q.shape[-1] == k_transpose_scaled.shape[-2] == v.shape[-1]
        and q.shape[-1] <= 160
    )
    fmha_requirements = q.shape[-1] <= 160 and (
        attn_mask is None or attn_mask is not None and attn_mask.dtype == float16
    )
    if type_match and fmha_requirements:
        return ops.attention(q, k_transpose_scaled, v, attn_mask, is_causal)

    qk = ops.matmul(q, k_transpose_scaled)
    if attn_mask is not None:
        if attn_mask.dtype.is_float():
            qk = qk + attn_mask
        elif attn_mask.dtype == boolean:
            neginfs = ops.full(qk.shape, value=qk.dtype.min_value, dtype=qk.dtype, device=qk.device)
            qk = ops.where(attn_mask, qk, neginfs)
        else:
            raise NotImplementedError('hidet: attention mask must be bool or float')
    out = ops.matmul(ops.softmax(qk, axis=-1), v)
    return out


@register_function(torch.gather)
def gather(x: Tensor, dim: int, index: Tensor, *, sparse_grad=False, out=None):
    if sparse_grad:
        warnings.warn_once('hidet: gather with sparse_grad=True is not supported. Treat as sparse_grad=False.')
    if out is not None:
        raise NotImplementedError('hidet: gather with out=... is not supported')
    return ops.gather(x, index, axis=dim)


@register_function(torch.maximum)
def maximum(x: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    a, b = x, other
    if len(a.shape) == 0 and a.device.is_cpu() and b.device.is_cuda() and not a.is_symbolic():
        a = a.cuda()
    if len(b.shape) == 0 and b.device.is_cpu() and a.device.is_cuda() and not b.is_symbolic():
        b = b.cuda()
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.maximum(..., out=...)")
    return ops.maximum(a, b)


@register_function(torch.minimum)
def minimum(x: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    a, b = x, other
    if len(a.shape) == 0 and a.device.is_cpu() and b.device.is_cuda() and not a.is_symbolic():
        a = a.cuda()
    if len(b.shape) == 0 and b.device.is_cpu() and a.device.is_cuda() and not b.is_symbolic():
        b = b.cuda()
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.minimum(..., out=...)")
    return ops.minimum(a, b)


@register_function(torch.max)
def torch_max(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.max(..., out=...)")
    return ops.max(x, dims=list(range(len(x.shape))), keep_dim=True)


@register_function(torch.max)
def torch_max(
    x: Tensor, other: Union[Tensor, int], *, out: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.max(..., out=...)")
    if isinstance(other, Tensor):
        return maximum(x, other)
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
def torch_min(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.min(..., out=...)")
    return ops.min(x, dims=list(range(len(x.shape))), keep_dim=True)


@register_function(torch.min)
def torch_min(
    x: Tensor, other: Union[Tensor, int], *, out: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.min(..., out=...)")
    if isinstance(other, Tensor):
        return minimum(x, other)
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


@register_function(operator.lt)
def lt(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a < b


@register_function(operator.le)
def le(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a <= b


@register_function(operator.gt)
def gt(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a > b


@register_function(operator.ge)
def ge(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a >= b


@register_function(operator.eq)
def eq(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        from hidet.graph.ops.utils import convert_to_tensor

        if isinstance(a, Tensor):
            return ops.equal(a, convert_to_tensor(b, a))
        else:
            return ops.equal(b, convert_to_tensor(a, b))
    return a == b


@register_function(operator.ne)
def ne(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a != b


@register_function(torch.rsqrt)
def rsqrt(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.rsqrt(..., out=...)")
    return ops.rsqrt(x)


@register_function(torch.pow)
@register_method(torch.Tensor.pow)
def tensor_pow(self: Union[Tensor, Number], exponent: Union[Tensor, Number]) -> Tensor:
    if isinstance(self, Tensor) and isinstance(exponent, Tensor):
        return ops.pow(self, exponent)
    elif isinstance(self, Tensor):
        return ops.pow(self, ops.full([], value=exponent, dtype=self.dtype, device=self.device))
    elif isinstance(exponent, Tensor):
        return ops.pow(ops.full([], value=self, dtype=exponent.dtype, device=exponent.device), exponent)
    else:
        return operator.pow(self, exponent)


@register_function(torch.mean)
@register_method(torch.Tensor.mean)
def torch_mean(x: Tensor, *, dtype: Optional[DataType] = None) -> Tensor:
    if dtype:
        x = x.astype(dtype_from_torch(dtype))
    output = ops.mean(x, dims=list(range(len(x.shape))), keep_dim=True)
    return output


@register_function(torch.mean)
@register_method(torch.Tensor.mean)
def torch_mean(
    x: Tensor, dim, keepdim=False, *, dtype: Optional[DataType] = None, out: Optional[Tensor] = None
) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.mean(..., out=...)")
    if dtype:
        x = x.astype(dtype_from_torch(dtype))
    output = ops.mean(x, dims=dim, keep_dim=keepdim)
    return output


@register_function(torch.sum)
@register_method(torch.Tensor.sum)
def torch_sum(x: Tensor, *, dtype: Optional[DataType] = None) -> Tensor:
    if dtype:
        x = x.astype(dtype_from_torch(dtype))
    output = ops.sum(x, dims=list(range(len(x.shape))), keep_dim=True)
    return output


@register_function(torch.sum)
@register_method(torch.Tensor.sum)
def torch_sum(
    x: Tensor, dim, keepdim=False, *, dtype: Optional[DataType] = None, out: Optional[Tensor] = None
) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.sum(..., out=...)")
    if dtype:
        x = x.astype(dtype_from_torch(dtype))
    output = ops.sum(x, dims=dim, keep_dim=keepdim)
    return output


@register_function(torch.cumsum)
def torch_cumsum(x: Tensor, dim, *, dtype: Optional[DataType] = None, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.cumsum(..., out=...)")
    if dtype:
        x = x.astype(dtype_from_torch(dtype))
    output = ops.cumsum(x, dim=dim)
    return output


@register_function(torch.ne)
@register_method(torch.Tensor.ne)
def torch_ne(x: Tensor, y: Union[Tensor, float, int], out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.ne(..., out=...)")
    if isinstance(y, (float, int)):
        y = ops.full(x.shape, y, dtype=x.dtype, device=x.device)
    output = ops.not_equal(x, y)
    return output


@register_function(torch.stack)
def torch_stack(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: int = 0, *, out: Optional[Tensor] = None):
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.stack(..., out=...)")
    tensors = [ops.unsqueeze(t, dims=dim) for t in tensors]
    return ops.concat(tensors, axis=dim)


@register_function(torch.conj)
@register_method(torch.Tensor.conj)
def torch_conj(x: Tensor) -> Tensor:
    if x.dtype.is_complex():
        return ops.conj(x)
    else:
        return x


@register_function(torch._C._log_api_usage_once)
@register_function(torch.cuda.synchronize)
def torch_noop(*args, **kwargs):
    return


@register_function(torch.abs)
def abs(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.abs(..., out=...)")
    return ops.abs(x)


@register_function(torch.log)
def log(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.log(..., out=...)")
    return ops.log(x)


@register_function(torch.full_like)
def full_like(
    x: Tensor,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=torch.preserve_format,
):
    if layout is not None:
        raise NotImplementedError("hidet: does not support torch.full(..., layout=..., ...)")

    hidet_device: Device = device_from_torch(torch_device=device) if device else x.device
    hidet_dtype: DataType = dtype_from_torch(torch_dtype=dtype) if dtype else x.dtype

    return ops.full(x.shape, fill_value, dtype=hidet_dtype, device=hidet_device)


@register_function(torch.zeros_like)
def zeros_like(
    x: Tensor, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format
):
    if layout is not None:
        raise NotImplementedError("layout is not None")

    hidet_device: Device = device_from_torch(torch_device=device) if device else x.device
    hidet_dtype: DataType = dtype_from_torch(torch_dtype=dtype) if dtype else x.dtype

    return ops.full(x.shape, dtype=hidet_dtype, device=hidet_device, value=hidet_dtype.zero)


@register_function(torch.clamp)
def clamp(
    x: Tensor,
    min: Optional[Union[Tensor, Number]] = None,
    max: Optional[Union[Tensor, Number]] = None,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.clamp(..., out=...)")

    min = convert_to_scalar_if_possible(min)
    max = convert_to_scalar_if_possible(max)

    if min is None and max is None:
        return x
    elif min is None:
        if not isinstance(max, Tensor):
            assert isinstance(max, (int, float, complex))
            max = ops.full([], value=max, dtype=x.dtype, device=x.device)
        return ops.minimum(x, max)
    elif max is None:
        if not isinstance(min, Tensor):
            assert isinstance(min, (int, float, complex))
            min = ops.full([], value=min, dtype=x.dtype, device=x.device)
        return ops.maximum(x, min)
    else:
        return ops.clamp(x, min, max)


@register_function(torch.isinf)
def isinf(x: Tensor) -> Tensor:
    return ops.isinf(x)


@register_function(torch.nn.functional.pad)
def torch_pad(x: Tensor, pad: Union[Tuple[int], List[int]], mode: str = 'constant', value=0):
    if isinstance(pad, tuple):
        pad = list(pad)
    # Torch's pad list has form [p2left, p2right, p1left, p1right, p0left, p0right]
    # Hidet's pad list has form [p0left, p1left, p2left, p0right, p1right, p2right]
    left = []
    right = []
    for i, p in enumerate(pad):
        if i % 2 == 0:
            left.append(p)
        else:
            right.append(p)
    left.reverse()
    right.reverse()
    pad = []
    for p in left:
        pad.append(p)
    for p in right:
        pad.append(p)
    return ops.pad(x, pads=pad, mode=mode, value=value)


@register_function(torch.roll)
def torch_roll(x: Tensor, shifts: Union[int, Sequence[int]], dims: Union[int, Sequence[int]] = None):
    return ops.roll(x, shifts, dims)


@register_function(torch.nn.functional.normalize)
def torch_normalize(x: Tensor, p=2.0, dim=1, eps=1e-12, out=None):
    if out is not None:
        raise NotImplementedError("out is not None")
    return ops.lp_norm(x, p, dim, eps)


@register_function(torch.clone)
@register_method(torch.Tensor.clone)
def torch_clone(x: Tensor, *, memory_format=torch.preserve_format):
    if memory_format is not torch.preserve_format:
        warnings.warn_once(
            "torch.clone got memory_format not torch.preserve_format, treating it as torch.preserve_format"
        )
    if x.is_symbolic():
        return x
    else:
        return x.copy()


@register_function(torch.chunk)
def torch_chunk(x: Tensor, chunks: int, dim: int = 0):
    return ops.split(x, parts_or_sections=chunks, axis=dim)


@register_function(torch.einsum)
def torch_einsum(equation, *operands):
    return ops.einsum(equation, operands)
