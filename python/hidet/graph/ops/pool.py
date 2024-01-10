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
from typing import Union, Sequence, List, Optional

from hidet.ir.expr import Expr, Int, convert, if_then_else, logical_and
from hidet.ir.dtypes import boolean, int32
from hidet.ir import primitives

from .utils import Task, Operator, Tensor, TensorNode, compute, reduce, input_like, normalize_stride, normalize_kernel
from .utils import normalize_padding, normalize_output
from ..transforms import ResolveRule, register_resolve_rule


class PoolNdBaseTask(Task):
    @staticmethod
    def preprocess(
        x: TensorNode,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]],
        ceil_mode: bool,
        channel_last: bool,
        reduce_type: str,
    ):
        assert len(x.shape) >= 3

        in_shape: List[Expr] = list(x.shape)
        if not channel_last:
            batch_dim: int = 0
            channel_dim: int = 1
            spatial_dims: List[int] = list(range(2, len(in_shape)))
        else:
            batch_dim: int = 0
            channel_dim: int = len(in_shape) - 1
            spatial_dims: List[int] = list(range(1, len(in_shape) - 1))

        kernel = normalize_kernel(kernel, dim=len(spatial_dims))
        stride = normalize_stride(stride, dim=len(spatial_dims))
        padding = normalize_padding(padding, dim=len(spatial_dims))

        # calculate output shape
        out_shape: List[Expr] = [int32.zero] * len(in_shape)
        out_shape[batch_dim] = in_shape[batch_dim]
        out_shape[channel_dim] = in_shape[channel_dim]
        for i, dim in enumerate(spatial_dims):
            if ceil_mode:
                out_shape[dim] = (
                    in_shape[dim] + padding[i] + padding[i + len(spatial_dims)] - kernel[i] + stride[i] - 1
                ) // stride[i] + 1
            else:
                out_shape[dim] = (in_shape[dim] + padding[i] + padding[i + len(spatial_dims)] - kernel[i]) // stride[
                    i
                ] + 1

        # calculate padding shape
        pad_shape: List[Expr] = [int32.zero] * len(in_shape)
        pad_shape[batch_dim] = in_shape[batch_dim]
        pad_shape[channel_dim] = in_shape[channel_dim]
        for i, dim in enumerate(spatial_dims):
            pad_shape[dim] = in_shape[dim] + padding[i] + padding[i + len(spatial_dims)]

        def f_pad_compute(*indices: Expr) -> Expr:
            if reduce_type == 'max':
                pad_value = x.type.dtype.min_value
            else:
                assert reduce_type == 'avg'
                pad_value = x.type.dtype.zero
            cond = boolean.true
            x_indices: List[Expr] = [int32.zero] * len(in_shape)
            x_indices[batch_dim] = indices[batch_dim]
            x_indices[channel_dim] = indices[channel_dim]
            for i, dim in enumerate(spatial_dims):
                cond = logical_and(cond, padding[i] <= indices[dim], indices[dim] < padding[i] + in_shape[dim])
                x_indices[dim] = indices[dim] - padding[i]
            return if_then_else(cond, x[x_indices], pad_value)

        pad = compute(name='pad', shape=pad_shape, fcompute=f_pad_compute)
        return kernel, stride, padding, batch_dim, channel_dim, spatial_dims, in_shape, out_shape, pad


class AvgPoolNdTask(PoolNdBaseTask):
    def __init__(
        self,
        x: TensorNode,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        channel_last: bool,
    ):
        kernel, stride, padding, batch_dim, channel_dim, spatial_dims, in_shape, out_shape, pad = self.preprocess(
            x, kernel, stride, padding, ceil_mode, channel_last, 'avg'
        )

        # calculate the sum of the pooling region
        def f_sum_compute(out_indices: List[Expr], reduce_indices: List[int]) -> Expr:
            pad_indices: List[Expr] = [int32.zero] * len(in_shape)
            pad_indices[batch_dim] = out_indices[batch_dim]
            pad_indices[channel_dim] = out_indices[channel_dim]
            for i, dim in enumerate(spatial_dims):
                pad_indices[dim] = out_indices[dim] * stride[i] + reduce_indices[i]
            return pad[pad_indices]

        s = compute(
            name='s',
            shape=out_shape,
            fcompute=lambda *out_indices: reduce(
                shape=kernel,
                fcompute=lambda *reduce_indices: f_sum_compute(out_indices, reduce_indices),
                reduce_type='sum',
            ),
        )

        # calculate the output value by dividing the sum by the number of pooling region elements
        def f_average_compute(*indices):
            if divisor_override is not None:
                area = int32(int(divisor_override))
            else:
                area = 1
                for i, dim in enumerate(spatial_dims):
                    if count_include_pad:
                        if ceil_mode:
                            start = indices[dim] * stride[i]
                            end = primitives.min(
                                indices[dim] * stride[i] + kernel[i],
                                convert(in_shape[dim] + padding[i] + padding[i + len(spatial_dims)]),
                            )
                            num_elements = end - start
                            area = area * num_elements
                        else:
                            area = area * kernel[i]
                    else:
                        start = primitives.max(indices[dim] * stride[i], convert(padding[i]))
                        end = primitives.min(indices[dim] * stride[i] + kernel[i], convert(in_shape[dim] + padding[i]))
                        num_elements = end - start
                        area = area * num_elements
            return s[indices] / area

        y = compute(name='y', shape=out_shape, fcompute=f_average_compute)
        super().__init__(
            name='max_pool{}d'.format(len(spatial_dims)),
            inputs=[x],
            outputs=[y],
            attributes={
                'kernel': kernel,
                'strides': stride,
                'padding': padding,
                'ceil_mode': ceil_mode,
                'count_include_pad': count_include_pad,
                'divisor_override': divisor_override,
                'channel_last': channel_last,
            },
        )


class MaxPoolNdTask(PoolNdBaseTask):
    def __init__(self, x: TensorNode, kernel, stride, padding, ceil_mode: bool, channel_last: bool):
        kernel, stride, padding, batch_dim, channel_dim, spatial_dims, in_shape, out_shape, pad = self.preprocess(
            x, kernel, stride, padding, ceil_mode, channel_last, 'max'
        )

        def f_compute(out_indices: List[Expr], reduce_indices: List[Expr]) -> Expr:
            pad_indices: List[Expr] = [int32.zero] * len(in_shape)
            pad_indices[batch_dim] = out_indices[batch_dim]
            pad_indices[channel_dim] = out_indices[channel_dim]
            for i, dim in enumerate(spatial_dims):
                pad_indices[dim] = out_indices[dim] * stride[i] + reduce_indices[i]
            return pad[pad_indices]

        y = compute(
            name='y',
            shape=out_shape,
            fcompute=lambda *out_indices: reduce(
                shape=kernel, fcompute=lambda *reduce_indices: f_compute(out_indices, reduce_indices), reduce_type='max'
            ),
        )
        super().__init__(name='max_pool{}d'.format(len(spatial_dims)), inputs=[x], outputs=[y])


class AdaptivePoolTask(Task):
    def __init__(self, x: TensorNode, output_size: Sequence[int], reduce_type: str):
        assert reduce_type in ['max', 'avg']
        x_shape: List[Expr] = x.shape  # [N, C, D1, D2, ...]
        output_size: List[Expr] = normalize_output(output_size, len(x_shape) - 2)
        y_shape: List[Int] = x_shape[:2] + output_size
        spatial_ndim = len(output_size)

        def start_index(y_indices: Sequence[Expr], spatial_dim: int) -> Expr:
            a = y_indices[spatial_dim + 2] * x_shape[spatial_dim + 2]
            b = y_shape[spatial_dim + 2]
            return a / b

        def end_index(y_indices: Sequence[Expr], spatial_dim: int) -> Expr:
            a = (1 + y_indices[spatial_dim + 2]) * x_shape[spatial_dim + 2]
            b = y_shape[spatial_dim + 2]
            return (a + b - 1) / b

        def grid_compute(*y_indices: Expr):
            start_indices: List[Expr] = [start_index(y_indices, dim) for dim in range(spatial_ndim)]
            end_indices: List[Expr] = [end_index(y_indices, dim) for dim in range(spatial_ndim)]
            reduce_shape: List[Expr] = [end_indices[dim] - start_indices[dim] for dim in range(spatial_ndim)]

            def reduce_compute(*reduce_indices: Expr) -> Expr:
                x_indices: List[Expr] = list(y_indices[:2])
                for dim in range(spatial_ndim):
                    x_indices.append(start_indices[dim] + reduce_indices[dim])
                return x[x_indices]

            return reduce(
                shape=reduce_shape, fcompute=reduce_compute, reduce_type=reduce_type, accumulate_dtype=x.type.dtype.name
            )

        y = compute(name='y', shape=y_shape, fcompute=grid_compute)
        super().__init__(
            name='adaptive_{}_pool{}d'.format(reduce_type, spatial_ndim),
            inputs=[x],
            outputs=[y],
            attributes={'output_size': output_size},
        )


class AdaptivePoolChannelLastTask(Task):
    def __init__(self, x: TensorNode, output_size: Sequence[int], reduce_type: str):
        assert reduce_type in ['max', 'avg']
        x_shape: List[Expr] = x.shape  # [N, D1, D2, ..., C]
        output_size: List[Expr] = normalize_output(output_size, len(x_shape) - 2)
        y_shape: List[Int] = [x_shape[0]] + output_size + [x_shape[-1]]
        spatial_ndim = len(output_size)

        def start_index(y_indices: Sequence[Expr], spatial_dim: int) -> Expr:
            a = y_indices[spatial_dim + 1] * x_shape[spatial_dim + 1]
            b = y_shape[spatial_dim + 1]
            return a / b

        def end_index(y_indices: Sequence[Expr], spatial_dim: int) -> Expr:
            a = (1 + y_indices[spatial_dim + 1]) * x_shape[spatial_dim + 1]
            b = y_shape[spatial_dim + 1]
            return (a + b - 1) / b

        def grid_compute(*y_indices: Expr):
            start_indices: List[Expr] = [start_index(y_indices, dim) for dim in range(spatial_ndim)]
            end_indices: List[Expr] = [end_index(y_indices, dim) for dim in range(spatial_ndim)]
            reduce_shape: List[Expr] = [end_indices[dim] - start_indices[dim] for dim in range(spatial_ndim)]

            def reduce_compute(*reduce_indices: Expr) -> Expr:
                x_indices: List[Expr] = [y_indices[0]]
                for dim in range(spatial_ndim):
                    x_indices.append(start_indices[dim] + reduce_indices[dim])
                x_indices.append(y_indices[-1])
                return x[x_indices]

            return reduce(
                shape=reduce_shape, fcompute=reduce_compute, reduce_type=reduce_type, accumulate_dtype=x.type.dtype.name
            )

        y = compute(name='y', shape=y_shape, fcompute=grid_compute)
        super().__init__(
            name='adaptive_{}_pool{}d_channel_last'.format(reduce_type, spatial_ndim),
            inputs=[x],
            outputs=[y],
            attributes={'output_size': output_size},
        )


class AvgPoolNdOp(Operator):
    ndim: Optional[int] = None
    last_channel: bool = False

    def __init__(
        self,
        x: Tensor,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]],
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ):
        if len(x.shape) != self.ndim + 2:
            raise ValueError(
                'AvgPool{}d expects {}D input, got {}D one.'.format(self.ndim, self.ndim + 2, len(x.shape))
            )

        super().__init__(
            inputs=[x],
            attributes={
                'kernel': kernel,
                'stride': stride,
                'padding': padding,
                'ceil_mode': ceil_mode,
                'count_include_pad': count_include_pad,
                'divisor_override': divisor_override,
            },
            task=AvgPoolNdTask(
                x=input_like(x, 'x'),
                kernel=kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
                channel_last=self.last_channel,
            ),
        )


class MaxPoolNdOp(Operator):
    ndim: Optional[int] = None
    last_channel: bool = False

    def __init__(
        self,
        x: Tensor,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]],
        ceil_mode: bool = False,
    ):
        if len(x.shape) != self.ndim + 2:
            raise ValueError(
                'MaxPool{}d expects {}D input, got {}D one.'.format(self.ndim, self.ndim + 2, len(x.shape))
            )

        super().__init__(
            inputs=[x],
            attributes={'kernel': kernel, 'stride': stride, 'padding': padding, 'ceil_mode': ceil_mode},
            task=MaxPoolNdTask(
                x=input_like(x, 'x'),
                kernel=kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                channel_last=self.last_channel,
            ),
        )


class AdaptivePoolNdOp(Operator):
    spatial_ndim: Optional[int] = None
    reduce_type: Optional[str] = None
    last_channel: Optional[bool] = None

    def __init__(self, x: Tensor, output_size):
        if len(x.shape) != self.spatial_ndim + 2:
            raise ValueError(
                'Adaptive{}Pool{}d expects {}D input, got {}D one.'.format(
                    self.reduce_type.capitalize(), self.spatial_ndim, self.spatial_ndim + 2, len(x.shape)
                )
            )
        output_size = normalize_output(output_size, self.spatial_ndim)
        self.reduce_type = self.reduce_type

        # todo: merge AdaptivePoolTask and AdaptivePoolChannelLastTask into one class
        if self.last_channel:
            task = AdaptivePoolChannelLastTask(input_like(x, 'x'), output_size, reduce_type=self.reduce_type)
        else:
            task = AdaptivePoolTask(input_like(x, 'x'), output_size, reduce_type=self.reduce_type)

        super().__init__(inputs=[x], attributes={'output_size': output_size}, task=task)


class MaxPool1dOp(MaxPoolNdOp):
    ndim: int = 1
    last_channel: bool = False


class MaxPool1dChannelLastOp(MaxPoolNdOp):
    ndim: int = 1
    last_channel: bool = True


class MaxPool2dOp(MaxPoolNdOp):
    ndim: int = 2
    last_channel: bool = False


class MaxPool2dChannelLastOp(MaxPoolNdOp):
    ndim: int = 2
    last_channel: bool = True


class MaxPool3dOp(MaxPoolNdOp):
    ndim: int = 3
    last_channel: bool = False


class MaxPool3dChannelLastOp(MaxPoolNdOp):
    ndim: int = 3
    last_channel: bool = True


class AvgPool1dOp(AvgPoolNdOp):
    ndim: int = 1
    last_channel: bool = False


class AvgPool1dChannelLastOp(AvgPoolNdOp):
    ndim: int = 1
    last_channel: bool = True


class AvgPool2dOp(AvgPoolNdOp):
    ndim: int = 2
    last_channel: bool = False


class AvgPool2dChannelLastOp(AvgPoolNdOp):
    ndim: int = 2
    last_channel: bool = True


class AvgPool3dOp(AvgPoolNdOp):
    ndim: int = 3
    last_channel: bool = False


class AvgPool3dChannelLastOp(AvgPoolNdOp):
    ndim: int = 3
    last_channel: bool = True


class AdaptiveAvgPool1dOp(AdaptivePoolNdOp):
    reduce_type = 'avg'
    spatial_ndim = 1
    last_channel = False


class AdaptiveAvgPool2dOp(AdaptivePoolNdOp):
    reduce_type = 'avg'
    spatial_ndim = 2
    last_channel = False


class AdaptiveAvgPool3dOp(AdaptivePoolNdOp):
    reduce_type = 'avg'
    spatial_ndim = 3
    last_channel = False


class AdaptiveAvgPool2dChannelLastOp(AdaptivePoolNdOp):
    reduce_type = 'avg'
    spatial_ndim = 2
    last_channel = True


class AdaptiveMaxPool1dOp(AdaptivePoolNdOp):
    reduce_type = 'max'
    spatial_ndim = 1
    last_channel = False


class AdaptiveMaxPool2dOp(AdaptivePoolNdOp):
    reduce_type = 'max'
    spatial_ndim = 2
    last_channel = False


class AdaptiveMaxPool3dOp(AdaptivePoolNdOp):
    reduce_type = 'max'
    spatial_ndim = 3
    last_channel = False


def max_pool1d(x: Tensor, kernel, stride, padding, ceil_mode=False) -> Tensor:
    return MaxPool1dOp(x, kernel, stride, padding, ceil_mode).outputs[0]


def max_pool2d(x: Tensor, kernel, stride, padding, ceil_mode=False) -> Tensor:
    return MaxPool2dOp(x, kernel, stride, padding, ceil_mode).outputs[0]


def max_pool3d(x: Tensor, kernel, stride, padding) -> Tensor:
    return MaxPool3dOp(x, kernel, stride, padding).outputs[0]


def avg_pool1d(
    x: Tensor,
    kernel,
    stride,
    padding,
    ceil_mode=False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> Tensor:
    return AvgPool1dOp(x, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override).outputs[0]


def avg_pool2d(
    x: Tensor,
    kernel,
    stride,
    padding,
    ceil_mode=False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> Tensor:
    return AvgPool2dOp(x, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override).outputs[0]


def avg_pool3d(
    x: Tensor,
    kernel,
    stride,
    padding,
    ceil_mode=False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> Tensor:
    return AvgPool3dOp(x, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override).outputs[0]


def adaptive_avg_pool1d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveAvgPool1dOp(x, output_size).outputs[0]


def adaptive_avg_pool2d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveAvgPool2dOp(x, output_size).outputs[0]


def adaptive_avg_pool3d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveAvgPool3dOp(x, output_size).outputs[0]


def adaptive_max_pool1d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveMaxPool1dOp(x, output_size).outputs[0]


def adaptive_max_pool2d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveMaxPool2dOp(x, output_size).outputs[0]


def adaptive_max_pool3d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveMaxPool3dOp(x, output_size).outputs[0]


# channel last operators
def max_pool1d_channel_last(x: Tensor, kernel, stride, padding, ceil_mode=False) -> Tensor:
    return MaxPool1dChannelLastOp(x, kernel, stride, padding, ceil_mode).outputs[0]


def max_pool2d_channel_last(x: Tensor, kernel, stride, padding, ceil_mode=False) -> Tensor:
    return MaxPool2dChannelLastOp(x, kernel, stride, padding, ceil_mode).outputs[0]


def max_pool3d_channel_last(x: Tensor, kernel, stride, padding, ceil_mode=False) -> Tensor:
    return MaxPool2dChannelLastOp(x, kernel, stride, padding, ceil_mode).outputs[0]


def avg_pool1d_channel_last(
    x: Tensor,
    kernel,
    stride,
    padding,
    ceil_mode=False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> Tensor:
    return AvgPool1dChannelLastOp(x, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override).outputs[0]


def avg_pool2d_channel_last(
    x: Tensor,
    kernel,
    stride,
    padding,
    ceil_mode=False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> Tensor:
    return AvgPool2dChannelLastOp(x, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override).outputs[0]


def avg_pool3d_channel_last(
    x: Tensor,
    kernel,
    stride,
    padding,
    ceil_mode=False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> Tensor:
    return AvgPool3dChannelLastOp(x, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override).outputs[0]


def adaptive_avg_pool2d_channel_last(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveAvgPool2dChannelLastOp(x, output_size).outputs[0]


@register_resolve_rule(AdaptivePoolNdOp)
class AdaptivePoolResolveRule(ResolveRule):
    def resolve(self, op: AdaptivePoolNdOp) -> Optional[List[Tensor]]:
        from hidet import ops

        assert isinstance(op, AdaptivePoolNdOp)
        x: Tensor = op.inputs[0]
        reduce_type = op.reduce_type
        if op.last_channel:
            spatial_dims = [i for i in range(1, len(x.shape) - 1)]
        else:
            spatial_dims = [i for i in range(2, len(x.shape))]

        if all(op.outputs[0].shape[i] == 1 for i in spatial_dims):

            if reduce_type == 'max':
                return [ops.max(x, dims=spatial_dims, keep_dim=True)]
            elif reduce_type == 'avg':
                return [ops.mean(x, dims=spatial_dims, keep_dim=True)]

        return None
