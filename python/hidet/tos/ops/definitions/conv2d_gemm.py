from typing import List

from .matmul import matmul
from .transform import pad
from .utils import Task, Operator, Tensor, compute, input_like, TensorInput, normalize_kernel, normalize_stride, normalize_padding


class Conv2dGemmImageTransformTask(Task):
    def __init__(self, x: TensorInput, kernel: List[int], stride: List[int]):
        n, c, h, w = x.const_shape()
        kx, ky = kernel
        sx, sy = stride
        p, q = (h - kx) // sx + 1, (w - ky) // sy + 1
        gemm_x = compute(
            name='gemm_x',
            shape=[n * p * q, c * kx * ky],
            fcompute=lambda i, k: x[i // (p * q), k // (kx * ky), i // q % p * sx + k // ky % kx, i % q * sy + k % ky],
            scope=x.data_type.scope
        )
        super().__init__(
            name='conv2d_gemm_image_transform',
            inputs=[x],
            outputs=[gemm_x]
        )


class Conv2dGemmFilterTransformTask(Task):
    def __init__(self, w: TensorInput):
        oc, c, kx, ky = w.const_shape()
        gemm_w = compute(
            name='gemm_w',
            shape=[c * kx * ky, oc],
            fcompute=lambda k, j: w[j, k // (kx * ky), k // ky % kx, k % ky],
            scope=w.data_type.scope
        )
        super().__init__(
            name='conv2d_gemm_filter_transform',
            inputs=[w],
            outputs=[gemm_w]
        )


class Conv2dGemmInverseTransformTask(Task):
    def __init__(self, gemm_y: TensorInput, out_shape: List[int]):
        n, oc, p, q = out_shape
        if tuple(gemm_y.const_shape()) != (n * p * q, oc):
            raise ValueError('Conv2d gemm inverse transform expect input with shape {}, got {}'.format(out_shape, gemm_y.const_shape()))
        y = compute(
            name='y',
            shape=[n, oc, p, q],
            fcompute=lambda i, j, r, s: gemm_y[i * (p * q) + r * q + s, j]
        )
        super().__init__(
            name='conv2d_gemm_inverse_transform',
            inputs=[gemm_y],
            outputs=[y]
        )


class Conv2dGemmImageTransformOp(Operator):
    def __init__(self, x: Tensor, kernel, stride):
        kernel = normalize_kernel(kernel)
        stride = normalize_stride(stride)
        super().__init__(
            inputs=[x],
            task=Conv2dGemmImageTransformTask(input_like(x, 'x'), kernel, stride),
            kernel=kernel,
            stride=stride
        )


class Conv2dGemmFilterTransformOp(Operator):
    def __init__(self, w: Tensor):
        super().__init__(
            inputs=[w],
            task=Conv2dGemmFilterTransformTask(input_like(w, 'w'))
        )


class Conv2dGemmInverseTransformOp(Operator):
    def __init__(self, gemm_y: Tensor, out_shape: List[int]):
        if len(out_shape) != 4:
            raise ValueError('Output shape expect with length 4, got {}'.format(out_shape))
        super().__init__(
            inputs=[gemm_y],
            task=Conv2dGemmInverseTransformTask(input_like(gemm_y, 'gemm_y'), out_shape)
        )


def conv2d_gemm_image_transform(x: Tensor, kernel: List[int], stride: List[int]) -> Tensor:
    return Conv2dGemmImageTransformOp(x, kernel, stride).get_output(0)


def conv2d_gemm_filter_transform(w: Tensor) -> Tensor:
    return Conv2dGemmFilterTransformOp(w).get_output(0)


def conv2d_gemm_inverse_transform(gemm_y: Tensor, out_shape: List[int]) -> Tensor:
    return Conv2dGemmInverseTransformOp(gemm_y, out_shape).get_output(0)


def conv2d_gemm(input: Tensor, weight: Tensor, padding, stride) -> Tensor:
    x = pad(input, normalize_padding(padding))
    gemm_x = conv2d_gemm_image_transform(x, kernel=weight.shape[2:], stride=stride)
    gemm_w = conv2d_gemm_filter_transform(weight)
    gemm_y = matmul(gemm_x, gemm_w)

    n, c, h, w = x.shape
    oc, c, kx, ky = weight.shape
    sx, sy = normalize_stride(stride)
    out_shape = [n, oc, (h - kx) // sx + 1, (w - ky) // sy + 1]
    y = conv2d_gemm_inverse_transform(gemm_y, out_shape)
    return y
