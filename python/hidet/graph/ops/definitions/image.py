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
from typing import Optional, List

from hidet.ir.expr import Expr, if_then_else, convert, cast, LogicalAnd, LogicalOr
from hidet.ir import primitives as prim
from .utils import Task, Operator, Tensor, TensorNode, compute, input_like


# Acknowledgement: take TVM resize topi implementation as a reference


def get_origin_index(x: Expr, image_width: int, target_width: int, coordinate_transformation_mode: str) -> Expr:
    scale = image_width / target_width
    func_map = {
        'half_pixel': lambda x: (x + 0.5) * scale - 0.5,
        'align_corners': lambda x: x * ((image_width - 1) / (target_width - 1)),
        'asymmetric': lambda x: x * scale,
        'pytorch_half_pixel': lambda x: (x + 0.5) * scale - 0.5 if target_width > 1 else convert(0.0),
        'tf_half_pixel_for_nn': lambda x: (x + 0.5) * scale,
    }
    if coordinate_transformation_mode not in func_map:
        raise ValueError(
            'Unsupported coordinate transformation mode: {}, candidates: {}.'.format(
                coordinate_transformation_mode, func_map.keys()
            )
        )
    return func_map[coordinate_transformation_mode](x)


def get_closest_index(x: Expr, rounding_method: str) -> Expr:
    func_map = {
        'round': lambda x: cast(prim.round(x), 'int32'),
        'round_prefer_floor': lambda x: cast(prim.ceil(x - 0.5), 'int32'),
        'round_prefer_ceil': lambda x: cast(prim.floor(x + 0.5), 'int32'),
        'floor': lambda x: cast(prim.floor(x + 1e-5), 'int32'),  # add epsilon (1e-5) to prevent gpu rounding error
        'ceil': lambda x: cast(prim.ceil(x - 1e-5), 'int32'),  # sub epsilon (1e-5) to prevent gpu rounding error
    }
    if rounding_method not in func_map:
        raise ValueError('Unsupported rounding_method: {}, candidates: {}'.format(rounding_method, func_map.keys()))
    return func_map[rounding_method](x)


def get_2d_pixel(data: TensorNode, n, c, h, w) -> Expr:
    height, width = data.const_shape()[2:]
    h = prim.max(0, prim.min(height - 1, h))
    w = prim.max(0, prim.min(width - 1, w))
    return data[n, c, h, w]


def linear_interpolate(a, b, ratio):
    return a * (1.0 - ratio) + b * ratio


def get_cubic_weights(s, a):
    # See equations (4)-(6) in https://ieeexplore.ieee.org/document/1163711
    s2 = s * s
    s3 = s * s * s
    w1 = a * (s3 - 2 * s2 + s)
    w2 = (a + 2) * s3 - (3 + a) * s2 + 1
    w3 = -(a + 2) * s3 + (3 + 2 * a) * s2 - a * s
    w4 = -a * s3 + a * s2
    return [w1, w2, w3, w4]


def cubic_interpolate(inputs, weights):
    return sum(inputs_i * weights_i for inputs_i, weights_i in zip(inputs, weights))


def resize2d_nchw_compute(
    data: TensorNode,
    size: List[int],
    method: str,
    coordinate_transformation_mode,
    rounding_method,
    roi,
    cubic_alpha,
    cubic_exclude,
    extrapolation_value,
):  # pylint: disable=unused-argument
    image_size = data.const_shape()[2:]
    target_size = size
    image_height = image_size[0]
    image_width = image_size[1]

    def fmap(n, c, h, w):
        h = get_origin_index(h, image_size[0], target_size[0], coordinate_transformation_mode)
        w = get_origin_index(w, image_size[1], target_size[1], coordinate_transformation_mode)
        if method == 'nearest':
            h = get_closest_index(h, rounding_method)
            w = get_closest_index(w, rounding_method)
            value = get_2d_pixel(data, n, c, h, w)
        elif method == 'linear':
            h_int = cast(prim.floor(h), 'int32')
            w_int = cast(prim.floor(w), 'int32')
            h_ratio = h - h_int
            w_ratio = w - w_int
            pixels = [[get_2d_pixel(data, n, c, h_int + i, w_int + j) for j in range(2)] for i in range(2)]
            top = linear_interpolate(*pixels[0], w_ratio)
            bottom = linear_interpolate(*pixels[1], w_ratio)
            value = linear_interpolate(top, bottom, h_ratio)
        elif method == 'cubic':
            h_int = cast(prim.floor(h), 'int32')
            w_int = cast(prim.floor(w), 'int32')
            h_ratio = h - prim.floor(h)
            w_ratio = w - prim.floor(w)
            pixels = [[get_2d_pixel(data, n, c, h_int + i - 1, w_int + j - 1) for j in range(4)] for i in range(4)]

            weight_w = get_cubic_weights(w_ratio, cubic_alpha)
            weight_h = get_cubic_weights(h_ratio, cubic_alpha)
            if cubic_exclude:
                for i in range(4):
                    weight_w[i] = if_then_else(
                        LogicalOr.join((w_int - 1 + i) < 0, (w_int + i) > image_width), 0.0, weight_w[i]
                    )
                    weight_h[i] = if_then_else(
                        LogicalOr.join((h_int - 1 + i) < 0, (h_int + i) > image_height), 0.0, weight_h[i]
                    )
                sum_weight_w = sum(weight_w)
                sum_weight_h = sum(weight_h)
                weight_w = [w / sum_weight_w for w in weight_w]
                weight_h = [h / sum_weight_h for h in weight_h]
            col0 = cubic_interpolate(pixels[0], weight_w)
            col1 = cubic_interpolate(pixels[1], weight_w)
            col2 = cubic_interpolate(pixels[2], weight_w)
            col3 = cubic_interpolate(pixels[3], weight_w)
            value = cubic_interpolate([col0, col1, col2, col3], weight_h)

        else:
            raise ValueError(
                'Unsupported scaling method: {}, candidates: {}'.format(method, ['nearest', 'linear', 'cubic'])
            )
        if coordinate_transformation_mode == 'tf_half_pixel_for_nn':
            value = if_then_else(
                LogicalAnd.join(0 <= h, h < image_size[0], 0 <= w, w < image_size[1]), value, extrapolation_value
            )
        return value

    output_shape = data.const_shape()[:2] + list(target_size)
    out = compute('out', shape=output_shape, fcompute=fmap)
    return out


class Resize2dTask(Task):
    def __init__(
        self,
        data: TensorNode,
        size: List[int],
        method: str,
        coordinate_transformation_mode,
        rounding_method,
        roi,
        cubic_alpha,
        cubic_exclude,
        extrapolation_value,
    ):
        out = resize2d_nchw_compute(
            data,
            size,
            method,
            coordinate_transformation_mode,
            rounding_method,
            roi,
            cubic_alpha,
            cubic_exclude,
            extrapolation_value,
        )
        super().__init__(name='resize2d', inputs=[data], outputs=[out])


class Resize2dOp(Operator):
    supported_methods = ['nearest', 'linear', 'cubic']
    supported_coord_trans_mode = [
        'half_pixel',
        'align_corners',
        'asymmetric',
        'pytorch_half_pixel',
        'tf_half_pixel_for_nn',
        'tf_crop_and_resize',
    ]
    supported_rounding_methods = ['round', 'round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil']

    def __init__(
        self,
        data,
        size: List[int],
        method: str,
        coordinate_transformation_mode: str,
        rounding_method: str,
        roi: Optional,
        cubic_alpha: Optional,
        cubic_exclude: Optional,
        extrapolation_value: Optional,
    ):
        if method not in self.supported_methods:
            raise ValueError("Resize only support methods: {}, but got {}.".format(self.supported_methods, method))
        if coordinate_transformation_mode not in self.supported_coord_trans_mode:
            raise ValueError(
                "Resize only support coordinate transformation modes: {}, but got {}.".format(
                    self.supported_coord_trans_mode, coordinate_transformation_mode
                )
            )
        if method == 'nearest' and rounding_method not in self.supported_rounding_methods:
            raise ValueError(
                "Resize only support rounding methods: {}, but got {}.".format(
                    self.supported_rounding_methods, rounding_method
                )
            )
        if len(size) != 2:
            raise ValueError('Resize2d expect size has 2 elements (height, width), got {}'.format(size))

        super().__init__(
            inputs=[data],
            task=Resize2dTask(
                input_like(data, 'data'),
                size,
                method,
                coordinate_transformation_mode,
                rounding_method,
                roi,
                cubic_alpha,
                cubic_exclude,
                extrapolation_value,
            ),
            attributes={
                'size': size,
                'method': method,
                'coordinate_transformation_mode': coordinate_transformation_mode,
                'rounding_method': rounding_method,
                'roi': roi,
                'cubic_alpha': cubic_alpha,
                'cubic_exclude': cubic_exclude,
                'extrapolation_value': extrapolation_value,
            },
        )


def resize2d(
    data: Tensor,
    size: List[int],
    method: str,
    coordinate_transformation_mode: str,
    rounding_method: str,
    roi: Optional,
    cubic_alpha: Optional,
    cubic_exclude: Optional,
    extrapolation_value: Optional,
) -> Tensor:
    return Resize2dOp(
        data,
        size,
        method,
        coordinate_transformation_mode,
        rounding_method,
        roi,
        cubic_alpha,
        cubic_exclude,
        extrapolation_value,
    ).get_output(0)
