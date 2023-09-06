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
from typing import Union, Sequence, List, Dict, Any, Optional

from hidet.ir.expr import Expr, Int, convert, if_then_else, logical_and

from .utils import Task, Operator, Tensor, TensorNode, compute, reduce, input_like, normalize_stride, normalize_kernel
from .utils import normalize_padding, normalize_output
from ..transforms import ResolveRule, register_resolve_rule


class Pool2dChannelLastTask(Task):
    def __init__(self, x: TensorNode, kernel, strides, padding, ceil_mode: bool, reduce_type: str):
        assert reduce_type in ['max', 'avg']
        kernel = normalize_kernel(kernel)
        strides = normalize_stride(strides)
        padding = normalize_padding(padding)
        batch_size, height, width, channels = x.shape
        if ceil_mode:
            out_height = (height + padding[0] + padding[2] - kernel[0] + strides[0] - 1) // strides[0] + 1
            out_width = (width + padding[1] + padding[3] - kernel[1] + strides[1] - 1) // strides[1] + 1
        else:
            out_height = (height + padding[0] + padding[2] - kernel[0]) // strides[0] + 1
            out_width = (width + padding[1] + padding[3] - kernel[1]) // strides[1] + 1
        pad_value = convert(0.0 if reduce_type == 'avg' else -1e30, dtype=x.type.dtype)
        pad = compute(
            name='pad',
            shape=[batch_size, height + padding[0] + padding[2], width + padding[1] + padding[3], channels],
            fcompute=lambda n, h, w, c: if_then_else(
                logical_and(padding[0] <= h, h < height + padding[0], padding[1] <= w, w < width + padding[1]),
                x[n, h - padding[0], w - padding[1], c],
                pad_value,
            ),
        )
        y = compute(
            name='y',
            shape=[batch_size, out_height, out_width, channels],
            fcompute=lambda n, h, w, c: reduce(
                shape=[kernel[0], kernel[1]],
                fcompute=lambda rx, ry: pad[n, h * strides[0] + rx, w * strides[1] + ry, c],
                reduce_type=reduce_type,
            ),
        )
        super().__init__(name='{}_pool2d_channel_last'.format(reduce_type), inputs=[x], outputs=[y])


class Pool2dTask(Task):
    def __init__(self, x: TensorNode, kernel, strides, padding, ceil_mode: bool, reduce_type: str):
        assert reduce_type in ['max', 'avg']
        kernel = normalize_kernel(kernel)
        strides = normalize_stride(strides)
        padding = normalize_padding(padding)
        batch_size, channels, height, width = x.shape
        if ceil_mode:
            out_height = (height + padding[0] + padding[2] - kernel[0] + strides[0] - 1) // strides[0] + 1
            out_width = (width + padding[1] + padding[3] - kernel[1] + strides[1] - 1) // strides[1] + 1
        else:
            out_height = (height + padding[0] + padding[2] - kernel[0]) // strides[0] + 1
            out_width = (width + padding[1] + padding[3] - kernel[1]) // strides[1] + 1
        pad_value = convert(0.0 if reduce_type == 'avg' else -1e30, dtype=x.type.dtype)
        pad = compute(
            name='pad',
            shape=[batch_size, channels, height + padding[0] + padding[2], width + padding[1] + padding[3]],
            fcompute=lambda n, c, h, w: if_then_else(
                logical_and(padding[0] <= h, h < height + padding[0], padding[1] <= w, w < width + padding[1]),
                x[n, c, h - padding[0], w - padding[1]],
                pad_value,
            ),
        )
        y = compute(
            name='y',
            shape=[batch_size, channels, out_height, out_width],
            fcompute=lambda n, c, h, w: reduce(
                shape=[kernel[0], kernel[1]],
                fcompute=lambda rx, ry: pad[n, c, h * strides[0] + rx, w * strides[1] + ry],
                reduce_type=reduce_type,
            ),
        )
        super().__init__(name='{}_pool2d'.format(reduce_type), inputs=[x], outputs=[y])


class Pool3dTask(Task):
    def __init__(self, x: TensorNode, kernel, strides, padding, reduce_type: str):
        assert reduce_type in ['max', 'avg']
        kernel = normalize_kernel(kernel, dim=3)
        strides = normalize_stride(strides, dim=3)
        padding = normalize_padding(padding, dim=3)
        batch_size, channels, depth, height, width = x.shape
        out_depth = (depth + padding[0] + padding[3] - kernel[0]) // strides[0] + 1
        out_height = (height + padding[1] + padding[4] - kernel[1]) // strides[1] + 1
        out_width = (width + padding[2] + padding[5] - kernel[2]) // strides[2] + 1
        pad_value = convert(0.0 if reduce_type == 'avg' else -1e30, dtype=x.type.dtype)
        pad = compute(
            name='pad',
            shape=[
                batch_size,
                channels,
                depth + padding[0] + padding[3],
                height + padding[1] + padding[4],
                width + padding[2] + padding[5],
            ],
            fcompute=lambda n, c, d, h, w: (
                if_then_else(
                    logical_and(
                        padding[0] <= d,
                        d < depth + padding[0],
                        padding[1] <= h,
                        h < height + padding[1],
                        padding[2] <= w,
                        w < width + padding[2],
                    ),
                    x[n, c, d - padding[0], h - padding[1], w - padding[2]],
                    pad_value,
                )
            ),
        )
        y = compute(
            name='y',
            shape=[batch_size, channels, out_depth, out_height, out_width],
            fcompute=lambda n, c, d, h, w: reduce(
                shape=[kernel[0], kernel[1], kernel[2]],
                fcompute=lambda rz, rx, ry: pad[n, c, d * strides[0] + rz, h * strides[1] + rx, w * strides[2] + ry],
                reduce_type=reduce_type,
            ),
        )
        super().__init__(name='{}_pool3d'.format(reduce_type), inputs=[x], outputs=[y])


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


class MaxPool2dOp(Operator):
    def __init__(
        self,
        x: Tensor,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]],
        ceil_mode: bool,
    ):
        super().__init__(
            inputs=[x],
            attributes={'kernel': kernel, 'stride': stride, 'padding': padding, 'ceil_mode': ceil_mode},
            task=Pool2dTask(input_like(x, 'x'), kernel, stride, padding, ceil_mode, reduce_type='max'),
        )


class MaxPool2dChannelLastOp(Operator):
    def __init__(
        self,
        x: Tensor,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]],
        ceil_mode: bool,
    ):
        super().__init__(
            inputs=[x],
            attributes={'kernel': kernel, 'stride': stride, 'padding': padding, 'ceil_mode': ceil_mode},
            task=Pool2dChannelLastTask(input_like(x, 'x'), kernel, stride, padding, ceil_mode, reduce_type='max'),
        )


class MaxPool3dOp(Operator):
    def __init__(
        self,
        x: Tensor,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]],
    ):
        super().__init__(
            inputs=[x],
            attributes={'kernel': kernel, 'stride': stride, 'padding': padding},
            task=Pool3dTask(input_like(x, 'x'), kernel, stride, padding, reduce_type='max'),
        )


class AvgPool2dOp(Operator):
    def __init__(
        self,
        x: Tensor,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]],
        ceil_mode: bool,
    ):
        super().__init__(
            inputs=[x],
            attributes={'kernel': kernel, 'stride': stride, 'padding': padding, 'ceil_mode': ceil_mode},
            task=Pool2dTask(input_like(x, 'x'), kernel, stride, padding, ceil_mode, reduce_type='avg'),
        )


class AvgPool2dChannelLastOp(Operator):
    def __init__(
        self,
        x: Tensor,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]],
        ceil_mode: bool,
    ):
        super().__init__(
            inputs=[x],
            attributes={'kernel': kernel, 'stride': stride, 'padding': padding, 'ceil_mode': ceil_mode},
            task=Pool2dChannelLastTask(input_like(x, 'x'), kernel, stride, padding, ceil_mode, reduce_type='avg'),
        )


class AvgPool3dOp(Operator):
    def __init__(
        self,
        x: Tensor,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int]],
    ):
        super().__init__(
            inputs=[x],
            attributes={'kernel': kernel, 'stride': stride, 'padding': padding},
            task=Pool3dTask(input_like(x, 'x'), kernel, stride, padding, reduce_type='avg'),
        )


class AdaptivePoolOp(Operator):
    def __init__(self, x: Tensor, output_size, reduce_type: str, attrs: Dict[str, Any], spatial_ndim: int):
        if len(x.shape) != spatial_ndim + 2:
            raise ValueError(
                'Adaptive{}Pool{}d expects {}D input, got {}D one.'.format(
                    reduce_type.capitalize(), spatial_ndim, spatial_ndim + 2, len(x.shape)
                )
            )
        output_size = normalize_output(output_size, spatial_ndim)
        self.reduce_type = reduce_type
        super().__init__(
            inputs=[x],
            attributes=attrs,
            task=AdaptivePoolTask(input_like(x, 'x'), output_size, reduce_type=reduce_type),
        )


class AdaptivePoolChannelLastOp(Operator):
    def __init__(self, x: Tensor, output_size, reduce_type: str, attrs: Dict[str, Any], spatial_ndim: int):
        if len(x.shape) != spatial_ndim + 2:
            raise ValueError(
                'Adaptive{}Pool{}d expects {}D input, got {}D one.'.format(
                    reduce_type.capitalize(), spatial_ndim, spatial_ndim + 2, len(x.shape)
                )
            )
        output_size = normalize_output(output_size, spatial_ndim)
        self.reduce_type = reduce_type
        super().__init__(
            inputs=[x],
            attributes=attrs,
            task=AdaptivePoolChannelLastTask(input_like(x, 'x'), output_size, reduce_type=reduce_type),
        )


class AdaptiveAvgPool1dOp(AdaptivePoolOp):
    def __init__(self, x: Tensor, output_size: Union[int, Sequence[int]]):
        super().__init__(x, output_size, reduce_type='avg', attrs={'output_size': output_size}, spatial_ndim=1)


class AdaptiveAvgPool2dOp(AdaptivePoolOp):
    def __init__(self, x: Tensor, output_size: Union[int, Sequence[int]]):
        super().__init__(x, output_size, reduce_type='avg', attrs={'output_size': output_size}, spatial_ndim=2)


class AdaptiveAvgPool2dChannelLastOp(AdaptivePoolChannelLastOp):
    def __init__(self, x: Tensor, output_size: Union[int, Sequence[int]]):
        super().__init__(x, output_size, reduce_type='avg', attrs={'output_size': output_size}, spatial_ndim=2)


class AdaptiveAvgPool3dOp(AdaptivePoolOp):
    def __init__(self, x: Tensor, output_size: Union[int, Sequence[int]]):
        super().__init__(x, output_size, reduce_type='avg', attrs={'output_size': output_size}, spatial_ndim=3)


class AdaptiveMaxPool1dOp(AdaptivePoolOp):
    def __init__(self, x: Tensor, output_size: Union[int, Sequence[int]]):
        super().__init__(x, output_size, reduce_type='max', attrs={'output_size': output_size}, spatial_ndim=1)


class AdaptiveMaxPool2dOp(AdaptivePoolOp):
    def __init__(self, x: Tensor, output_size: Union[int, Sequence[int]]):
        super().__init__(x, output_size, reduce_type='max', attrs={'output_size': output_size}, spatial_ndim=2)


class AdaptiveMaxPool3dOp(AdaptivePoolOp):
    def __init__(self, x: Tensor, output_size: Union[int, Sequence[int]]):
        super().__init__(x, output_size, reduce_type='max', attrs={'output_size': output_size}, spatial_ndim=3)


def max_pool2d(x: Tensor, kernel, stride, padding, ceil_mode=False) -> Tensor:
    return MaxPool2dOp(x, kernel, stride, padding, ceil_mode).outputs[0]


def max_pool2d_channel_last(x: Tensor, kernel, stride, padding, ceil_mode=False) -> Tensor:
    return MaxPool2dChannelLastOp(x, kernel, stride, padding, ceil_mode).outputs[0]


def max_pool3d(x: Tensor, kernel, stride, padding) -> Tensor:
    return MaxPool3dOp(x, kernel, stride, padding).outputs[0]


def avg_pool2d(x: Tensor, kernel, stride, padding, ceil_mode=False) -> Tensor:
    return AvgPool2dOp(x, kernel, stride, padding, ceil_mode).outputs[0]


def avg_pool2d_channel_last(x: Tensor, kernel, stride, padding, ceil_mode=False) -> Tensor:
    return AvgPool2dChannelLastOp(x, kernel, stride, padding, ceil_mode).outputs[0]


def avg_pool3d(x: Tensor, kernel, stride, padding) -> Tensor:
    return AvgPool3dOp(x, kernel, stride, padding).outputs[0]


def adaptive_avg_pool1d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveAvgPool1dOp(x, output_size).outputs[0]


def adaptive_avg_pool2d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveAvgPool2dOp(x, output_size).outputs[0]


def adaptive_avg_pool2d_channel_last(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveAvgPool2dChannelLastOp(x, output_size).outputs[0]


def adaptive_avg_pool3d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveAvgPool3dOp(x, output_size).outputs[0]


def adaptive_max_pool1d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveMaxPool1dOp(x, output_size).outputs[0]


def adaptive_max_pool2d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveMaxPool2dOp(x, output_size).outputs[0]


def adaptive_max_pool3d(x: Tensor, output_size: Union[int, Sequence[int]]) -> Tensor:
    return AdaptiveMaxPool3dOp(x, output_size).outputs[0]


@register_resolve_rule(AdaptivePoolOp)
class AdaptivePoolResolveRule(ResolveRule):
    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, AdaptivePoolOp)
        x: Tensor = op.inputs[0]
        output_size = op.attrs['output_size']
        reduce_type = op.reduce_type
        resolve_to_reduce = output_size == 1 if isinstance(output_size, int) else all(d == 1 for d in output_size)
        if resolve_to_reduce:
            dims = [i for i in range(len(x.shape))]
            from hidet.graph.ops import mean, max

            if reduce_type == 'max':
                return [max(x, dims=dims[2:], keep_dim=True)]
            elif reduce_type == 'avg':
                return [mean(x, dims=dims[2:], keep_dim=True)]
        return None
