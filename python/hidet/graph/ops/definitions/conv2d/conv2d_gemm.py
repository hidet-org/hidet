from typing import List

from hidet.graph.ops.definitions.matmul.batch_matmul import batch_matmul
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, compute, input_like, TensorNode
from hidet.graph.ops.definitions.utils import normalize_kernel, normalize_stride
from .utils import infer_conv2d_shape


class Conv2dGemmImageTransformTask(Task):
    def __init__(self, x: TensorNode, kernel: List[int], stride: List[int], groups: int):
        n, c, h, w = x.const_shape()
        kx, ky = kernel
        sx, sy = stride
        p, q = (h - kx) // sx + 1, (w - ky) // sy + 1
        if c % groups != 0:
            raise ValueError('Conv2d expect in_channels % groups == 0, but got in_channels {} and groups {}'.format(c, groups))
        gc = c // groups  # group channels
        gemm_x = compute(
            name='gemm_x',
            shape=[groups, n * p * q, gc * kx * ky],
            fcompute=lambda g, i, k: x[i // (p * q), g * gc + k // (kx * ky), i // q % p * sx + k // ky % kx, i % q * sy + k % ky]
        )
        super().__init__(
            name='conv2d_gemm_image_transform',
            inputs=[x],
            outputs=[gemm_x],
        )


# class Conv2dGemmFilterTransformTask(Task):
#     def __init__(self, w: TensorNode, groups: int):
#         oc, c, kx, ky = w.const_shape()
#         if oc % groups != 0 or c % groups != 0:
#             raise ValueError('Conv2d expects in_channels % groups == 0, out_channels % groups == 0, got {}, {}, {}'.format(c, oc, groups))
#         ogc = oc // groups  # out group channels
#         gemm_w = compute(
#             name='gemm_w',
#             shape=[groups, c * kx * ky, ogc],
#             fcompute=lambda g, k, j: w[g * ogc + j, k // (kx * ky), k // ky % kx, k % ky],
#             scope=w.data_type.scope
#         )
#         super().__init__(
#             name='conv2d_gemm_filter_transform',
#             inputs=[w],
#             outputs=[gemm_w]
#         )
#
#
# class Conv2dGemmInverseTransformTask(Task):
#     def __init__(self, gemm_y: TensorNode, out_shape: List[int]):
#         n, oc, p, q = out_shape
#         y_shape = gemm_y.const_shape()  # [groups, n * p * q, ogc]
#         groups = y_shape[0]
#
#         assert y_shape[-1] * y_shape[0] == oc
#         ogc = oc // groups
#
#         if tuple(y_shape) != (groups, n * p * q, ogc):
#             raise ValueError('Conv2d gemm inverse transform expect input with shape {}, got {}'.format(
#                 (groups, n * p * q, ogc), gemm_y.const_shape()))
#
#         y = compute(
#             name='y',
#             shape=[n, oc, p, q],
#             fcompute=lambda i, j, r, s: gemm_y[j // ogc, i * (p * q) + r * q + s, j % ogc],
#             scope=gemm_y.data_type.scope
#         )
#         super().__init__(
#             name='conv2d_gemm_inverse_transform',
#             inputs=[gemm_y],
#             outputs=[y],
#             inverse_map={
#                 gemm_y: lambda i, j: [i // (p * q), j, i // q % p, i % q]
#             }
#         )


class Conv2dGemmImageTransformOp(Operator):
    def __init__(self, x: Tensor, kernel, stride, groups):
        kernel = normalize_kernel(kernel)
        stride = normalize_stride(stride)
        super().__init__(
            inputs=[x],
            task=Conv2dGemmImageTransformTask(input_like(x, 'x'), kernel, stride, groups),
            attributes={
                'kernel': kernel,
                'stride': stride
            }

        )


#
# class Conv2dGemmFilterTransformOp(Operator):
#     def __init__(self, w: Tensor, groups):
#         super().__init__(
#             inputs=[w],
#             task=Conv2dGemmFilterTransformTask(input_like(w, 'w'), groups)
#         )
#
#
# class Conv2dGemmInverseTransformOp(Operator):
#     def __init__(self, gemm_y: Tensor, out_shape: List[int]):
#         if len(out_shape) != 4:
#             raise ValueError('Output shape expect with length 4, got {}'.format(out_shape))
#         super().__init__(
#             inputs=[gemm_y],
#             task=Conv2dGemmInverseTransformTask(input_like(gemm_y, 'gemm_y'), out_shape)
#         )
#


def conv2d_gemm_image_transform(x: Tensor, kernel: List[int], stride: List[int], groups: int = 1) -> Tensor:
    return Conv2dGemmImageTransformOp(x, kernel, stride, groups).get_output(0)


def conv2d_gemm_filter_transform(w: Tensor, groups: int = 1) -> Tensor:
    # weight shape: [oc, c, kx, ky]
    # output shape: [groups, c * kx * ky, ogc] where ogc = oc // groups
    oc, c, kx, ky = w.shape
    if oc % groups != 0:
        raise ValueError('invalid conv2d groups {} for out channels {}'.format(groups, oc))
    ogc = oc // groups
    w = w.reshape([groups, ogc, c, kx, ky])  # [groups, ogc, c, kx, ky]
    w = w.rearrange([[0], [2, 3, 4], [1]])  # [groups, c * kx * ky, ogc]
    return w


def conv2d_gemm_inverse_transform(gemm_y: Tensor, out_height, out_width) -> Tensor:
    # gemm_y shape: [groups, n * p * q, ogc]
    # output shape: [n, oc, p, q] where oc = groups * ogc
    p, q = out_height, out_width
    groups, npq, ogc = gemm_y.shape
    assert npq % (p * q) == 0
    n = npq // (p * q)
    y = gemm_y.reshape([groups, n, p, q, ogc])
    y = y.rearrange([[1], [0, 4], [2], [3]])
    return y


def conv2d_gemm(data: Tensor, weight: Tensor, stride, groups: int = 1) -> Tensor:
    gemm_x = conv2d_gemm_image_transform(data, kernel=weight.shape[2:], stride=stride, groups=groups)
    gemm_w = conv2d_gemm_filter_transform(weight, groups=groups)
    gemm_y = batch_matmul(gemm_x, gemm_w)

    y_shape = infer_conv2d_shape(data.shape, weight.shape, stride, groups)
    y = conv2d_gemm_inverse_transform(gemm_y, out_height=y_shape[2], out_width=y_shape[3])
    return y
