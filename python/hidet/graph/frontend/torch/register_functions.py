# pylint: disable=too-many-lines
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
import math
import operator
import functools
import torch

from hidet.graph.tensor import Tensor, from_torch, ones_like, randn
from hidet.graph import ops
from hidet.utils import same_list
from hidet.ir.type import DataType
from hidet.ir import expr
from hidet.ir.dtypes import promote_type
from hidet.ir.expr import Expr, Int, is_constant
from hidet.runtime.device import Device
from .registry import register_function, register_method
from .interpreter import warnings
from .utils import dtype_from_torch, device_from_torch, normalize_to_scalar, convert_to_scalar_if_possible

Number = Union[int, float, bool]


@register_function(torch.nn.functional.conv1d)
def conv1d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride=1, padding=0, dilation=1, groups=1):
    x = ops.conv_pad(x, padding)
    y = ops.conv1d(x, weight, stride=stride, dilations=dilation, groups=groups)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2])
    return y


@register_function(torch.nn.functional.conv_transpose1d)
def conv1d_transpose(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    if dilation != 1 and not same_list(dilation, [1]):
        raise NotImplementedError("dilation != 1")
    y = ops.conv1d_transpose(x, weight, stride, padding, groups, output_padding)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2])
    return y


@register_function(torch.nn.functional.conv2d)
def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride=1, padding=0, dilation=1, groups=1):
    y = ops.conv2d(x, weight, stride, dilation, groups, padding=padding)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2, 3])
    return y


@register_function(torch.nn.functional.conv_transpose2d)
def conv2d_transpose(
    x: Tensor, weight: Tensor, bias: Optional[Tensor], stride=1, padding=0, output_padding=0, groups=1, dilation=1
):
    if dilation != 1 and not same_list(dilation, [1, 1]):
        raise NotImplementedError("dilation != 1")
    y = ops.conv2d_transpose(x, weight, stride, padding, groups, output_padding)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2, 3])
    return y


@register_function(torch.nn.functional.conv3d)
def conv3d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride=1, padding=0, dilation=1, groups=1):
    x = ops.conv_pad(x, padding)
    y = ops.conv3d(x, weight, stride, dilation, groups)
    if bias is not None:
        y = y + ops.unsqueeze(bias, [0, 2, 3, 4])
    return y


@register_function(torch.nn.functional.conv_transpose3d)
def conv3d_transpose(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
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
@register_function(torch.relu)
@register_function(torch.relu_)
@register_method(torch.Tensor.relu)
@register_method(torch.Tensor.relu_)
def relu(x: Tensor, inplace: bool = False):
    # if inplace:
    #     warnings.warn_once('hidet: relu with inplace=True is not supported. Treat as inplace=False.')
    _ = inplace
    return ops.relu(x)


@register_function(torch.nn.functional.leaky_relu)
def leaky_relu(x: Tensor, negative_slope: float = 0.01, inplace: bool = False):
    _ = inplace
    return ops.leaky_relu(x, alpha=negative_slope)


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
def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None, weight_is_transposed=False):

    if len(weight.shape) > 1 and not weight_is_transposed:
        if weight.dtype.is_any_float16():
            y = ops.matmul_nt(x, weight)
        else:
            weight = ops.transpose(weight, [1, 0])
            y = ops.matmul(x, weight)
    else:
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
@register_function(torch.add)
@register_method(torch.Tensor.add)
@register_method(torch.Tensor.add_)
def add(x: Tensor, y: Tensor):
    return x + y


@register_function(operator.iadd)
def iadd(x: Tensor, y: Tensor):
    return x + y


@register_function(operator.imul)
def imul(x: Tensor, y: Tensor):
    return x * y


@register_function(torch.sin)
@register_function(torch.ops.aten.sin.default)
@register_function(torch.sin_)
@register_method(torch.Tensor.sin)
@register_method(torch.Tensor.sin_)
def sin(x: Tensor):
    return ops.sin(x)


@register_function(torch.cos)
@register_function(torch.ops.aten.cos.default)
@register_function(torch.cos_)
@register_method(torch.Tensor.cos)
@register_method(torch.Tensor.cos_)
def cos(x: Tensor):
    return ops.cos(x)


@register_function(operator.not_)
def not_(x: Union[Tensor, Expr]):
    if isinstance(x, Tensor):
        # not x when len(x) > 1 is not supported as it
        # results in 'RuntimeError: bool value of Tensor with more than one value is ambiguous'
        # return ops.logical_not(x)
        assert len(x.shape) <= 1, "not x when len(x) > 1 is not supported"
        return not x.item()
    elif isinstance(x, Expr):
        return expr.logical_not(x)
    else:
        return not x


@register_function(torch.logical_not)
@register_method(torch.Tensor.logical_not)
@register_method(torch.Tensor.logical_not_)
def logical_not(x: Tensor):
    return ops.logical_not(x)


@register_function(operator.and_)
@register_function(torch.bitwise_and)
@register_function(torch.ops.aten.bitwise_and.Tensor)
@register_method(torch.Tensor.bitwise_and)
@register_method(torch.Tensor.bitwise_and_)
def and_(x: Union[Tensor, Expr], y: Union[Tensor, Expr]):
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return ops.bitwise_and(x, y)
    else:
        # TODO: Should this also be changed to bitwise_and for Expr?
        return expr.bitwise_and(x, y)


@register_function(torch.logical_and)
@register_method(torch.Tensor.logical_and)
@register_method(torch.Tensor.logical_and_)
def logical_and(x: Tensor, y: Tensor):
    return ops.logical_and(x, y)


@register_function(torch.logical_or)
@register_method(torch.Tensor.logical_or)
@register_method(torch.Tensor.logical_or_)
def logical_or(x: Tensor, y: Tensor):
    return ops.logical_or(x, y)


@register_function(operator.or_)
@register_function(torch.bitwise_or)
@register_function(torch.ops.aten.bitwise_or.Tensor)
@register_method(torch.Tensor.bitwise_or)
@register_method(torch.Tensor.bitwise_or_)
def or_(x: Union[Tensor, Expr], y: Union[Tensor, Expr]):
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return ops.bitwise_or(x, y)
    else:
        return expr.bitwise_or(x, y)


@register_function(operator.xor)
@register_function(torch.bitwise_xor)
@register_function(torch.ops.aten.bitwise_xor.Tensor)
@register_method(torch.Tensor.bitwise_xor)
@register_method(torch.Tensor.bitwise_xor_)
def xor(x: Union[Tensor, Expr], y: Union[Tensor, Expr]):
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return ops.bitwise_xor(x, y)
    else:
        return expr.bitwise_xor(x, y)


@register_function(torch.logical_xor)
@register_method(torch.Tensor.logical_xor)
@register_method(torch.Tensor.logical_xor_)
def logical_xor(x: Tensor, y: Tensor):
    return ops.logical_xor(x, y)


@register_function(operator.invert)
@register_function(torch.bitwise_not)
@register_method(torch.Tensor.bitwise_not)
@register_method(torch.Tensor.bitwise_not_)
def invert(x: Tensor):
    from hidet import boolean

    if x.dtype is boolean:
        return ops.logical_not(x)
    else:
        return ops.bitwise_invert(x)


@register_function(torch.nn.functional.batch_norm)
def batch_norm(
    x: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05,
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
@register_method(torch.Tensor.flatten)
def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1):
    return ops.flatten(x, start_dim, end_dim)


@register_function(operator.getitem)
def getitem(x: Tensor, index):
    if isinstance(index, Tensor):
        if x.device != index.device:
            if index.device.kind == 'cpu':
                index = index.to(device=x.device)
            else:
                raise NotImplementedError(
                    'getitem: index tensor must be either on CPU or the same device as the tensor'
                )
    elif isinstance(index, (tuple, list)):
        index = list(index)
        for i, v in enumerate(index):
            if isinstance(v, Tensor) and x.device != v.device:
                if v.device.kind == 'cpu':
                    index[i] = v.to(device=x.device)
                else:
                    raise NotImplementedError(
                        'getitem: index tensor must be either on CPU or the same device as the tensor'
                    )
    return x[index]


@register_method(torch.Tensor.item)
def torch_tensor_item(x: Tensor):
    assert len(x.shape) == 1 and x.shape[0] == 1, "item() only supports 1-element tensor"
    return x[0]


@register_function(operator.setitem)
def setitem(x: Tensor, item, setvalue):
    if isinstance(item, (list, tuple)) and all(isinstance(i, (list, tuple)) for i in item):
        # ((1, 1, 5), (2, 3, 6)) -> ((1, 2), (1, 3), (5, 6))
        items = tuple(zip(*item))
        for i in items:
            x = setitem(x, i, setvalue)
        return x

    if isinstance(item, list):
        item = tuple(item)
    if not isinstance(item, tuple):
        item = tuple([item])

    if isinstance(setvalue, Tensor):
        if x.device != setvalue.device:
            # turns out setvalue can be on any device, and it will be moved to x.device
            setvalue = ops.transfer(setvalue, x.device)
        if setvalue.dtype != x.dtype:
            setvalue = ops.cast(setvalue, x.dtype)

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

    if isinstance(setvalue, Tensor):
        squeeze_dims = [i for i, dimlen in enumerate(setvalue.shape) if dimlen == 1]
        setvalue = ops.squeeze(setvalue, squeeze_dims)

    out = ops.set_strided_slice(x, setvalue, starts, ends, steps)
    return out


@register_function(operator.mul)
@register_function(torch.mul)
@register_function(torch.multiply)
@register_function(torch.ops.aten.mul.Tensor)
@register_method(torch.Tensor.mul)
@register_method(torch.Tensor.mul_)
@register_method(torch.Tensor.multiply)
@register_method(torch.Tensor.multiply_)
def mul(x: Tensor, y: Tensor):
    return x * y


@register_function(torch.cat)
def cat(tensors: List[Tensor], dim: int = 0):
    dtype = functools.reduce(promote_type, [t.dtype for t in tensors])
    tensors = [ops.cast(t, dtype) for t in tensors]
    return ops.concat(tensors, dim)


@register_function(torch.cat)
def cat_v2(tensors: List[Tensor], axis: int):  # PyTorch supports axis as well as the argument name
    dtype = functools.reduce(promote_type, [t.dtype for t in tensors])
    tensors = [ops.cast(t, dtype) for t in tensors]
    return ops.concat(tensors, axis)


@register_function(torch.squeeze)
@register_method(torch.Tensor.squeeze)
@register_method(torch.Tensor.squeeze_)
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


@register_function(torch.unsqueeze)
@register_method(torch.Tensor.unsqueeze)
@register_method(torch.Tensor.unsqueeze_)
def unsqueeze(x: Tensor, dim: int):
    dim = int(dim)
    dim = dim if dim >= 0 else dim + len(x.shape) + 1
    return ops.unsqueeze(x, [dim])


@register_function(torch.nn.functional.avg_pool2d)
def avg_pool2d(
    x: Tensor, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None
):
    if stride is None:
        stride = kernel_size
    y = ops.avg_pool2d(
        x,
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    return y


@register_function(torch.nn.functional.avg_pool3d)
def avg_pool3d(x: Tensor, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None):
    y = ops.avg_pool3d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
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


@register_function(operator.itruediv)
@register_function(operator.truediv)
@register_function(torch.true_divide)
@register_method(torch.Tensor.true_divide)
@register_method(torch.Tensor.true_divide_)
def truediv(x: Union[Tensor, int, float], y: Union[Tensor, int, float]):
    import hidet

    def is_integer(v: Union[Tensor, int, float]) -> bool:
        return isinstance(v, int) or (isinstance(v, Tensor) and v.dtype.is_integer())

    if not isinstance(x, Tensor) and not isinstance(y, Tensor):
        return x / y
    if is_integer(x) and is_integer(y):
        if isinstance(y, (int, float)):
            y = hidet.asarray(y).to(device=x.device)
        return x / ops.cast(y, 'float32')
    else:
        return x / y


@register_function(torch.div)
@register_function(torch.divide)
@register_method(torch.Tensor.div)
@register_method(torch.Tensor.div_)
@register_method(torch.Tensor.divide)
@register_method(torch.Tensor.divide_)
def div(x: Union[Tensor, Number], y: Union[Tensor, Number], *, rounding_mode: Optional[str] = None, out=None):
    result = truediv(x, y)
    if rounding_mode is None:
        return result
    elif rounding_mode == 'floor':
        # Turns out the `floor` rounding mode will follow a slightly different type promotion rule,
        # and this subtle difference is the cause of issue #264.
        from hidet import int32, float32

        primitive_type_map = {int: int32, float: float32, bool: int32}

        x_dtype = x.dtype if isinstance(x, Tensor) else primitive_type_map[type(x)]
        y_dtype = y.dtype if isinstance(y, Tensor) else primitive_type_map[type(y)]

        result = ops.floor(result)

        # `rounding_mode = 'floor'` retains the integer type if both inputs are integers
        if x_dtype.is_integer() and y_dtype.is_integer():
            return result.to(dtype=promote_type(x_dtype, y_dtype))
        else:
            return result

    else:
        assert rounding_mode == 'trunc', 'rounding_mode should be one of "floor" or "trunc"'
        if isinstance(result, Tensor):
            dtype = result.dtype
            result = result.to(dtype='int64')
            return result.to(dtype=dtype)
        else:
            if isinstance(x, float) or isinstance(y, float):
                return float(int(result))
            return int(result)


@register_function(torch.floor_divide)
@register_method(torch.Tensor.floor_divide)
@register_method(torch.Tensor.floor_divide_)
@register_function(operator.floordiv)
def floor_divide(x: Union[Tensor, Number], y: Union[Tensor, Number]):
    return div(x, y, rounding_mode='floor')


@register_function(torch.as_strided)
@register_method(torch.Tensor.as_strided)
def torch_as_strided(
    input: Tensor, size: Union[int, Tuple[int]], stride: Union[int, Tuple[int]], storage_offset: Optional[int] = None
):
    return ops.as_strided(input, size, stride, storage_offset)


@register_function(operator.sub)
@register_function(torch.sub)
@register_function(torch.subtract)
@register_function(torch.ops.aten.sub.Tensor)
@register_method(torch.Tensor.sub)
@register_method(torch.Tensor.sub_)
@register_method(torch.Tensor.subtract)
@register_method(torch.Tensor.subtract_)
def sub(x: Tensor, y: Tensor):
    return x - y


@register_function(operator.neg)
@register_function(torch.neg)
@register_function(torch.negative)
@register_function(torch.ops.aten.neg.default)
@register_method(torch.Tensor.neg)
@register_method(torch.Tensor.neg_)
@register_method(torch.Tensor.negative)
@register_method(torch.Tensor.negative_)
def neg(x: Tensor):
    return -x


@register_function(torch.nn.functional.softmax)
@register_method(torch.Tensor.softmax)
@register_function(torch.softmax)
def softmax(x: Tensor, dim: int, _stacklevel: int = 3, dtype=None):
    if dtype is not None:
        x = ops.cast(x, dtype_from_torch(dtype))
    return ops.softmax(x, dim)


@register_function(torch.nn.functional.log_softmax)
@register_method(torch.Tensor.log_softmax)
@register_function(torch.log_softmax)
def logsoftmax(x: Tensor, dim: int, _stacklevel: int = 3, dtype=None):
    if dtype is not None:
        x = ops.cast(x, dtype_from_torch(dtype))
    return ops.logsoftmax(x, dim)


@register_function(operator.matmul)
@register_function(torch.matmul)
@register_method(torch.Tensor.matmul)
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
@register_function(torch.tanh_)
@register_function(torch.nn.functional.tanh)
@register_method(torch.Tensor.tanh)
@register_method(torch.Tensor.tanh_)
def tanh(x: Tensor):
    return ops.tanh(x)


@register_function(torch.nn.functional.hardtanh)
def hardtanh(x: Tensor, min_val: float, max_val: float, inplace: bool = False):
    return ops.hardtanh(x, min_val, max_val, inplace=inplace)


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


@register_function(torch.nn.functional.embedding_bag)
def torch_embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Optional[Tensor] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    mode: str = 'mean',
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
):
    # Since we assume max_norm is None for now, norm_type is not used.
    # And we can ignore `sparse` in inference since it's about gradient.
    _, _ = norm_type, sparse  # unused

    if scale_grad_by_freq:
        raise NotImplementedError("scale_grad_by_freq=True is not supported for embedding_bag")
    if per_sample_weights is not None:
        raise NotImplementedError("per_sample_weights is not supported for embedding_bag")
    if max_norm is not None:
        raise NotImplementedError("max_norm is not supported for embedding_bag")
    if include_last_offset:
        raise NotImplementedError("include_last_offset is not supported for embedding_bag")
    if padding_idx is not None:
        raise NotImplementedError("padding_idx is not supported for embedding_bag")
    if mode not in ('sum', 'mean'):
        # TODO: Currently don't support 'max' since it is not encountered yet.
        assert mode == 'max'
        raise ValueError("embedding bag: mode 'max' is not supported yet")

    assert offsets is not None, "embedding_bag: currently we only support 1d inputs with offsets"

    return ops.embedding_bag(input, weight, offsets, mode=mode)


@register_function(torch.permute)
@register_method(torch.Tensor.permute)
def permute(x: Tensor, *args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    dims = [int(v) for v in args]
    return ops.transpose(x, dims)


@register_function(torch.repeat_interleave)
@register_method(torch.Tensor.repeat_interleave)
def repeat_interleave(
    x: Tensor, repeats: Union[int, Tensor], dim: Optional[int] = None, *, output_size: Optional[int] = None
):
    if output_size is not None:
        raise NotImplementedError("hidet: repeat_interleave with out_size is not supported yet")
    if isinstance(repeats, Tensor):
        raise NotImplementedError("hidet: repeat_interleave with Tensor repeats is not supported yet")
    if repeats <= 0:
        raise ValueError("repeat_interleave: repeats must be positive")
    return ops.repeat_interleave(x, repeats, dim=dim)


@register_function(torch.swapaxes)
@register_function(torch.transpose)
@register_method(torch.Tensor.transpose)
@register_method(torch.Tensor.swapaxes)
@register_method(torch.Tensor.swapaxes_)
@register_method(torch.Tensor.swapdims)
@register_method(torch.Tensor.swapdims_)
def transpose(x: Tensor, dim0: int, dim1: int):
    if dim0 < dim1:
        dim0, dim1 = dim1, dim0
    return ops.transpose(x, [dim0, dim1])


@register_function(torch.reshape)
def reshape(x: Tensor, shape: Tuple[int]):
    return ops.reshape(x, shape)


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
@register_method(torch.Tensor.addmm)
@register_method(torch.Tensor.addmm_)
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


@register_function(torch.index_select)
@register_method(torch.Tensor.index_select)
def index_select(self: Tensor, dim: int, index: Tensor):
    return ops.index_select(self, index, dim)


@register_function(torch.where)
def where(condition: Tensor, x: Union[Tensor, Number], y: Union[Tensor, Number]):
    return ops.where(cond=condition, x=x, y=y)


@register_method(torch.Tensor.where)
def tensor_where(self: Tensor, condition: Tensor, y: Union[Tensor, Number]):
    return ops.where(cond=condition, x=self, y=y)


@register_function(torch.pow)
@register_method(torch.Tensor.pow)
@register_method(torch.Tensor.pow_)
def torch_pow(base: Union[Number, Tensor], exponent: Union[Number, Tensor]):
    if isinstance(exponent, (int, float, bool)):
        if exponent in (2, 2.0):
            return ops.square(base)
        exponent = full_like(base, exponent)
    elif isinstance(base, (int, float, bool)):
        base = full_like(exponent, base)
    return ops.pow(base, exponent)


@register_function(torch.scalar_tensor)
def scalar_tensor(value):
    return ops.full([1], value)


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
    return ops.full(
        size, dtype=hidet_dtype, device=hidet_device, value=hidet_dtype.zero if hidet_dtype is not None else 0
    )


@register_function(torch.empty_like)
def empty_like(input, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    if layout is not None:
        raise NotImplementedError("hidet: does not support torch.empty_like(..., layout=..., ...)")
    if requires_grad and torch.is_grad_enabled():
        warnings.warn_once("hidet: requires_grad=True when torch.is_grad_enabled(), treating as requires_grad=False")
    if memory_format != torch.preserve_format:
        raise NotImplementedError("hidet: does not support torch.empty_like(..., memory_format=..., ...)")

    hidet_device: Device = device_from_torch(torch_device=device) if device is not None else input.device
    hidet_dtype: DataType = dtype_from_torch(torch_dtype=dtype) if dtype is not None else input.dtype
    return ops.full(input.shape, dtype=hidet_dtype, device=hidet_device, value=hidet_dtype.zero)


@register_function(torch.bmm)
@register_method(torch.Tensor.bmm)
def bmm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.bmm(..., out=...)")
    return ops.matmul(input, mat2)


@register_function(torch.baddbmm)
@register_method(torch.Tensor.baddbmm)
@register_method(torch.Tensor.baddbmm_)
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


@register_function(torch.as_tensor)
def torch_as_tensor(data: Any, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> Tensor:
    if isinstance(data, Tensor):
        device = device_from_torch(torch_device=device) if device is not None else device
        return data.to(device=device, dtype=dtype_from_torch(dtype))
    else:
        tt = torch.as_tensor(data, dtype=dtype, device=device)
        return from_torch(tt)


@register_function(torch.randn)
def torch_randn(*size, generator=None, out=None, layout=None, device=None, requires_grad=False, pin_memory=False):
    if generator is not None:
        raise NotImplementedError("hidet: currently does not support torch.randn(..., generator=..., ...)")
    if out is not None:
        raise NotImplementedError("hidet: currently does not support torch.randn(..., out=..., ...)")
    if layout is not None:
        raise NotImplementedError("hidet: currently does not support torch.randn(..., layout=..., ...)")
    if pin_memory:
        raise NotImplementedError("hidet: currently does not support torch.randn(..., pin_memory=True, ...)")
    if requires_grad and torch.is_grad_enabled():
        warnings.warn_once("hidet: requires_grad=True when torch.is_grad_enabled(), treating as requires_grad=False")

    device = device_from_torch(torch_device=device) if device is not None else device

    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]
    return randn(size, device=device)


@register_function(torch.randn_like)
def torch_randn_like(input, generator=None, out=None, layout=None, device=None, requires_grad=False, pin_memory=False):
    return torch_randn(
        input.shape,
        generator=generator,
        out=out,
        layout=layout,
        device=device if device is not None else input.device,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
    )


@register_function(torch.sigmoid)
@register_function(torch.sigmoid_)
@register_function(torch.nn.functional.sigmoid)
@register_method(torch.Tensor.sigmoid)
@register_method(torch.Tensor.sigmoid_)
def sigmoid(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        warnings.warn_once("hidet: does not support torch.sigmoid(..., out=...)")
    return ops.sigmoid(x)


@register_function(torch.exp)
@register_method(torch.Tensor.exp)
@register_method(torch.Tensor.exp_)
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


@register_function(torch.nn.functional.glu)
def glu(x: Tensor, dim: int = -1):

    # split the tensor into two halves along the specified dim
    x1, x2 = ops.split(x, 2, axis=dim)
    return x1 * ops.sigmoid(x2)


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
@register_method(torch.Tensor.hardshrink)
def hardshrink(x: Tensor, lambd=0.5):
    return ops.hardshrink(x, lambd)


@register_function(torch.nn.functional.softsign)
def softsign(x: Tensor):
    return ops.softsign(x)


@register_function(torch.nn.functional.celu)
@register_function(torch.celu)
@register_function(torch.celu_)
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
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
):
    from hidet import boolean, float32

    if attn_mask is not None and attn_mask.dtype == float32:
        attn_mask = attn_mask.to(q.dtype)

    if not math.isclose(dropout_p, 0.0):
        warnings.warn_once('hidet: attention dropout is not supported. Treat as dropout_p=0.0')

    k_rank = len(k.shape)
    if scale is None:
        scale = 1 / math.sqrt(k.shape[-1])
    # transpose last 2 dimensions of k, and normalize by sqrt(head_dim)
    k_transpose_scaled = ops.transpose(k, [i for i in range(k_rank - 2)] + [k_rank - 1, k_rank - 2]) * scale

    type_match = (
        q.dtype == k.dtype == v.dtype
        and q.dtype.is_any_float16()
        and len(q.shape) == len(k_transpose_scaled.shape) == len(v.shape)
        and k_transpose_scaled.shape[-1] == v.shape[-2]
        and q.shape[-1] == k_transpose_scaled.shape[-2] == v.shape[-1]
        and q.shape[-1] <= 160
    )
    fmha_requirements = q.shape[-1] <= 160 and (
        attn_mask is None or attn_mask is not None and attn_mask.dtype.is_any_float16()
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
@register_method(torch.Tensor.gather)
def gather(x: Tensor, dim: int, index: Tensor, *, sparse_grad=False, out=None):
    if sparse_grad:
        warnings.warn_once('hidet: gather with sparse_grad=True is not supported. Treat as sparse_grad=False.')
    if out is not None:
        raise NotImplementedError('hidet: gather with out=... is not supported')
    return ops.gather(x, index, axis=dim)


@register_function(torch.maximum)
@register_method(torch.Tensor.maximum)
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
@register_method(torch.Tensor.minimum)
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
@register_method(torch.Tensor.max)
def torch_max(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.max(..., out=...)")

    return ops.max(x, dims=list(range(len(x.shape))), keep_dim=False)


@register_function(torch.max)
@register_method(torch.Tensor.max)
def torch_max_v2(
    x: Tensor, other: Union[Tensor, int], *, out: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.max(..., out=...)")
    if isinstance(other, Tensor):
        return maximum(x, other)
    else:
        return torch_max_v3(x, other)


@register_function(torch.max)
@register_method(torch.Tensor.max)
def torch_max_v3(
    x: Tensor, dim: Int, keepdim: bool = False, *, out: Optional[Union[Tensor, Tuple[Tensor, ...], List[Tensor]]] = None
) -> Tuple[Tensor, Tensor]:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.max(..., out=...)")
    values = ops.max(x, dims=dim, keep_dim=keepdim)
    indices = ops.argmax(x, dim=dim, keep_dim=keepdim)
    return values, indices


@register_function(torch.min)
@register_method(torch.Tensor.min)
def torch_min(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.min(..., out=...)")

    return ops.min(x, dims=list(range(len(x.shape))), keep_dim=False)


@register_function(torch.min)
@register_method(torch.Tensor.min)
def torch_min_v2(
    x: Tensor, other: Union[Tensor, int], *, out: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.min(..., out=...)")
    if isinstance(other, Tensor):
        return minimum(x, other)
    else:
        return torch_min_v3(x, other)


@register_function(torch.min)
@register_method(torch.Tensor.min)
def torch_min_v3(
    x: Tensor, dim: Int, keepdim: bool = False, *, out: Optional[Union[Tensor, Tuple[Tensor, ...], List[Tensor]]] = None
) -> Tuple[Tensor, Tensor]:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.min(..., out=...)")
    values = ops.min(x, dims=dim, keep_dim=keepdim)
    indices = ops.argmin(x, dim=dim, keep_dim=keepdim)
    return values, indices


@register_function(operator.lt)
@register_function(torch.lt)
@register_function(torch.less)
@register_method(torch.Tensor.lt)
@register_method(torch.Tensor.lt_)
@register_method(torch.Tensor.less)
@register_method(torch.Tensor.less_)
def lt(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a < b


@register_function(operator.le)
@register_function(torch.le)
@register_function(torch.less_equal)
@register_method(torch.Tensor.le)
@register_method(torch.Tensor.le_)
@register_method(torch.Tensor.less_equal)
@register_method(torch.Tensor.less_equal_)
def le(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a <= b


@register_function(operator.gt)
@register_function(torch.gt)
@register_function(torch.greater)
@register_method(torch.Tensor.gt)
@register_method(torch.Tensor.gt_)
@register_method(torch.Tensor.greater)
@register_method(torch.Tensor.greater_)
def gt(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a > b


@register_function(operator.ge)
@register_function(torch.ge)
@register_function(torch.greater_equal)
@register_method(torch.Tensor.ge)
@register_method(torch.Tensor.ge_)
@register_method(torch.Tensor.greater_equal)
@register_method(torch.Tensor.greater_equal_)
def ge(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a >= b


@register_function(operator.eq)
@register_function(torch.eq)
@register_method(torch.Tensor.eq)
@register_method(torch.Tensor.eq_)
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


@register_function(operator.mod)
def mod(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a % b


@register_function(operator.lshift)
@register_function(torch.bitwise_left_shift)
@register_method(torch.Tensor.bitwise_left_shift)
@register_method(torch.Tensor.bitwise_left_shift_)
def lshift(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a << b


@register_function(operator.rshift)
@register_function(torch.bitwise_right_shift)
@register_method(torch.Tensor.bitwise_right_shift)
@register_method(torch.Tensor.bitwise_right_shift_)
def rshift(a: Union[Tensor, Expr, Number], b: Union[Tensor, Expr, Number]) -> Tensor:
    return a >> b


@register_function(torch.rsqrt)
@register_method(torch.Tensor.rsqrt)
@register_method(torch.Tensor.rsqrt_)
def rsqrt(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.rsqrt(..., out=...)")
    return ops.rsqrt(x)


@register_function(torch.sqrt)
@register_method(torch.Tensor.sqrt)
@register_method(torch.Tensor.sqrt_)
@register_function(torch._C._te.sqrt)
def sqrt(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.sqrt(..., out=...)")
    return ops.sqrt(x)


@register_function(operator.pow)
@register_function(torch.pow)
@register_method(torch.Tensor.pow)
@register_method(torch.Tensor.pow_)
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

    # turns out here keep_dim should be False too, similar to torch.max/min/sum
    output = ops.mean(x, dims=list(range(len(x.shape))), keep_dim=False)
    return output


@register_function(torch.mean)
@register_method(torch.Tensor.mean)
def torch_mean_v2(
    x: Tensor, dim, keepdim=False, *, dtype: Optional[DataType] = None, out: Optional[Tensor] = None
) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.mean(..., out=...)")
    if dtype:
        x = x.astype(dtype_from_torch(dtype))
    output = ops.mean(x, dims=dim, keep_dim=keepdim)
    return output


@register_function(torch.var)
@register_method(torch.Tensor.var)
def torch_var(
    x: Tensor,
    dim: Union[int, Tuple[int]],
    *,
    unbiased: bool = True,
    correction: int = 1,
    dtype: Optional[DataType] = None,
    keepdim: bool = False,
) -> Tensor:
    if dtype:
        x = x.astype(dtype_from_torch(dtype))

    if not unbiased:
        correction = 0
    output = ops.var(x, dims=dim, keep_dim=keepdim, correction=correction)
    return output


@register_function(torch.sum)
@register_method(torch.Tensor.sum)
def torch_sum(x: Tensor, *, dtype: Optional[DataType] = None) -> Tensor:
    if dtype:
        x = x.astype(dtype_from_torch(dtype))
    output = ops.sum(x, dims=list(range(len(x.shape))), keep_dim=False)
    return output


@register_function(torch.sum)
@register_method(torch.Tensor.sum)
def torch_sum_v2(
    x: Tensor, dim, keepdim=False, *, dtype: Optional[DataType] = None, out: Optional[Tensor] = None
) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.sum(..., out=...)")
    if dtype:
        x = x.astype(dtype_from_torch(dtype))
    output = ops.sum(x, dims=dim, keep_dim=keepdim)
    return output


@register_function(torch.cumsum)
@register_method(torch.Tensor.cumsum)
@register_method(torch.Tensor.cumsum_)
def torch_cumsum(x: Tensor, dim, *, dtype: Optional[DataType] = None, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.cumsum(..., out=...)")
    if dtype:
        x = x.astype(dtype_from_torch(dtype))
    output = ops.cumsum(x, dim=dim)
    return output


@register_function(torch.ne)
@register_function(torch.not_equal)
@register_method(torch.Tensor.ne)
@register_method(torch.Tensor.ne_)
@register_method(torch.Tensor.not_equal)
@register_method(torch.Tensor.not_equal_)
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
@register_function(torch._assert_async)
@register_function(torch.cuda.synchronize)
@register_function(torch.amp.autocast_mode._enter_autocast)
@register_function(torch.amp.autocast_mode._exit_autocast)
@register_function(torch._C._set_grad_enabled)
@register_function(torch.autograd.function.FunctionCtx)
def torch_noop(*args, **kwargs):
    return


@register_function(torch.abs)
@register_function(torch.abs_)
@register_function(torch.absolute)
@register_method(torch.Tensor.abs)
@register_method(torch.Tensor.abs_)
@register_method(torch.Tensor.absolute)
@register_method(torch.Tensor.absolute_)
def abs(x: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.abs(..., out=...)")
    return ops.abs(x)


@register_function(torch.log)
@register_function(torch.log_)
@register_method(torch.Tensor.log)
@register_method(torch.Tensor.log_)
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
@register_function(torch.clip)
@register_method(torch.Tensor.clamp)
@register_method(torch.Tensor.clamp_)
@register_method(torch.Tensor.clip)
@register_method(torch.Tensor.clip_)
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
@register_method(torch.Tensor.isinf)
def isinf(x: Tensor) -> Tensor:
    return ops.isinf(x)


@register_function(torch._C._nn.pad)
@register_function(torch.nn.functional.pad)
def torch_pad(x: Tensor, pad: Union[Tuple[int, ...], List[int]], mode: str = 'constant', value=None):
    if value is None:
        value = 0.0
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
@register_method(torch.Tensor.roll)
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


@register_method(torch.Tensor.copy_)
def torch_copy(x: Tensor, src: Tensor, non_blocking: bool = False):
    if non_blocking:
        warnings.warn_once("torch.Tensor.copy_ with non_blocking=True is not supported. Treating as non_blocking=False")
    if x.shape != src.shape:
        src = ops.broadcast(src, x.shape)
    return torch_clone(src)


@register_function(torch.chunk)
@register_method(torch.Tensor.chunk)
def torch_chunk(x: Tensor, chunks: int, dim: int = 0):
    dim_size = x.shape[dim]
    chunk_size = math.ceil(dim_size / chunks)
    parts = []
    for start in range(0, dim_size, chunk_size):
        parts.append(min(chunk_size, dim_size - start))
    assert sum(parts) == x.shape[dim]
    return ops.split(x, axis=dim, parts_or_sections=parts)


@register_function(torch.split)
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
    return ops.split(self, axis=dim, parts_or_sections=parts)


@register_function(torch.unbind)
@register_method(torch.Tensor.unbind)
def torch_unbind(x: Tensor, dim: int = 0):
    num_chunks = x.shape[dim]
    chunks = torch_chunk(x, num_chunks, dim=dim)
    return tuple(chunk.squeeze(dim) for chunk in chunks)


@register_function(torch.einsum)
def torch_einsum(equation, *operands):
    return ops.einsum(equation, operands)


@register_function(torch.triu)
@register_method(torch.Tensor.triu)
@register_method(torch.Tensor.triu_)
def torch_triu(x: Tensor, diagonal: int = 0, *, out=None):
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.triu(..., out=...)")
    return ops.triu(x, diagonal)


@register_function(torch.tril)
@register_method(torch.Tensor.tril)
@register_method(torch.Tensor.tril_)
def torch_tril(x: Tensor, diagonal: int = 0, *, out=None):
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.tril(..., out=...)")
    return ops.tril(x, diagonal)


@register_function(torch.meshgrid)
def torch_meshgrid(*tensors, indexing='ij'):
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tensors[0]
    return ops.meshgrid(tensors, indexing=indexing)


@register_function(torch.all)
def torch_all(input):
    return ops.all(input, axis=None, keepdims=False)


@register_function(torch.all)
def torch_all_v2(input, dim: Union[int, Sequence[int]], keepdim=False, *, out=None):
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.all(..., out=...)")
    if isinstance(dim, int):
        dim = (dim,)
    return ops.all(input, axis=dim, keepdims=keepdim)


@register_function(torch.ones_like)
def torch_ones_like(
    x: Tensor, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format
):
    dtype = dtype_from_torch(dtype) if dtype is not None else dtype_from_torch(x.dtype)
    device = device_from_torch(device) if device is not None else device_from_torch(x.device)
    return ones_like(x, dtype=dtype, device=device)


@register_function(torch.argmax)
@register_method(torch.Tensor.argmax)
def torch_argmax(x, dim: Int = None, keepdim: bool = False):
    return ops.argmax(x, dim, keepdim)


@register_function(torch.argmin)
@register_method(torch.Tensor.argmin)
def torch_argmin(x, dim: Int = None, keepdim: bool = False):
    return ops.argmin(x, dim, keepdim)


@register_function(torch.any)
def torch_any_v1(input: Tensor, dim: Union[int, Sequence[int]], keepdim=False, *, out=None) -> Tensor:
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.any(..., out=...)")
    if isinstance(dim, int):
        dim = (dim,)
    return ops.any(input, axis=dim, keepdims=keepdim)


@register_function(torch.any)
def torch_any_v2(input: Tensor) -> Tensor:
    return ops.any(input)


@register_function(torch.t)
@register_method(torch.Tensor.t)
def torch_t(input: Tensor):
    assert 0 <= len(input.shape) <= 2, 'torch.t expects tensors <= 2D'
    if len(input.shape) == 2:
        return ops.transpose(input, [1, 0])
    return input


@register_function(torch.nn.functional.unfold)
def torch_unfold(input: Tensor, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    assert 3 <= len(input.shape) <= 4, "torch.nn.functional.unfold accepts 3D or 4D tensor only"
    return ops.im2col(input, kernel_size, dilation, padding, stride)


@register_method(torch.Tensor.scatter_)
def torch_scatter_(input: Tensor, dim: int, index: Tensor, src: Tensor, reduce: str = None) -> Tensor:
    if reduce is None:
        reduce = 'replace'
    return ops.scatter_(input, dim, index, src, reduce)


@register_method(torch.Tensor.scatter_add_)
def torch_scatter_add_(input: Tensor, dim: int, index: Tensor, src: Tensor):
    return ops.scatter_add_(input, dim, index, src)


@register_function(torch.flip)
@register_method(torch.Tensor.flip)
def torch_unfold(input: Tensor, dims) -> Tensor:
    return ops.flip(input, dims)


@register_function(torch.sign)
def torch_sign(input: Tensor, *, out=None):
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.sign(..., out=...)")
    return ops.sign(input)


@register_function(torch.ceil)
def torch_ceil(input: Tensor, *, out=None):
    if out is not None:
        raise NotImplementedError("hidet: does not support torch.ceil(..., out=...)")
    return ops.ceil(input)


@register_function(torch.cuda.nccl.all_reduce)
@register_function(torch._C._nccl_all_reduce)
@register_function(torch.distributed.all_reduce)
@register_function(torch.ops._c10d_functional.all_reduce)
@register_function("torch.ops.vllm.all_reduce")
def torch_all_reduce(tensor: Tensor, op_name='sum', group_name=None):
    from hidet.distributed import is_initialized, init_process_group, set_nccl_comms
    from hidet.cuda.nccl.ffi import load_nccl_library

    if not is_initialized():
        load_nccl_library()
        init_process_group(
            backend='nccl',
            init_method='file:///tmp/hidet-nccl-init-group',
            world_size=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
        )
        set_nccl_comms()
    res = ops.all_reduce(tensor, op_name)
    return res


@register_function(torch.ops._c10d_functional.wait_tensor)
def torch_wait_tensor(x: Tensor):
    return ops.wait_tensor(x)


@register_function(getattr)
def torch_getattr(obj, attr):
    if isinstance(obj, Tensor) and attr == 'is_cpu':
        return obj.device.kind == 'cpu'
    raise RuntimeError('Unsupported getattr')


# Below torch function might appear in fxgraph on dynamo level. But dynamo resolved it by itself.
# Hidet never should see them.
@register_function(torch._C._has_torch_function)
@register_function(torch._C._has_torch_function_unary)
@register_function(torch._C._has_torch_function_variadic)
@register_function(torch._C._get_tracing_state)
@register_function(torch._jit_internal.is_scripting)
@register_function(torch.are_deterministic_algorithms_enabled)
@register_function(torch._C._is_tracing)
@register_function(torch.jit._trace.is_tracing)
@register_function(torch._dynamo.external_utils.is_compiling)
@register_function(torch._utils.is_compiling)
@register_function(torch.compiler.is_compiling)
@register_function(torch.compiler.is_dynamo_compiling)
@register_function(torch._assert)
@register_function(torch._assert_scalar)
@register_function(torch._assert_tensor_metadata)
@register_function(math.acos)
@register_function(math.acosh)
@register_function(math.asin)
@register_function(math.asinh)
@register_function(math.atan)
@register_function(math.atan2)
@register_function(math.atanh)
@register_function(math.ceil)
@register_function(math.comb)
@register_function(math.copysign)
@register_function(math.cos)
@register_function(math.cosh)
@register_function(math.degrees)
@register_function(math.dist)
@register_function(math.erf)
@register_function(math.erfc)
@register_function(math.exp)
@register_function(math.expm1)
@register_function(math.fabs)
@register_function(math.factorial)
@register_function(math.floor)
@register_function(math.fmod)
@register_function(math.frexp)
@register_function(math.fsum)
@register_function(math.gamma)
@register_function(math.gcd)
@register_function(math.hypot)
@register_function(math.isclose)
@register_function(math.isfinite)
@register_function(math.isinf)
@register_function(math.isnan)
@register_function(math.isqrt)
# @register_function(math.lcm)
@register_function(math.ldexp)
@register_function(math.lgamma)
@register_function(math.log)
@register_function(math.log10)
@register_function(math.log1p)
@register_function(math.log2)
@register_function(math.modf)
# @register_function(math.nextafter)
@register_function(math.perm)
@register_function(math.pow)
@register_function(math.prod)
@register_function(math.radians)
@register_function(math.remainder)
@register_function(math.sin)
@register_function(math.sinh)
@register_function(math.sqrt)
@register_function(math.tan)
@register_function(math.tanh)
@register_function(math.trunc)
# @register_function(math.ulp)
def torch_should_not_appear_function(*args, **kwargs):
    raise RuntimeError('These function should not apper in fxgraph')
