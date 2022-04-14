from typing import List

import numpy as np
import pytest

import hidet.tos.operators as ops
from hidet.testing import check_binary


def numpy_conv2d(data: np.ndarray, weight: np.ndarray, padding: List[int], stride: List[int]):
    n, c, h, w = data.shape
    oc, _, kx, ky = weight.shape
    padded_shape = [n, c, padding[0] + h + padding[2], padding[1] + w + padding[3]]
    padded_data = np.zeros_like(data, shape=padded_shape)
    padded_data[:, :, padding[0]:padding[0] + h, padding[1]:padding[1] + w] = data
    oh, ow = [(v - k) // s + 1 for v, k, s in zip(padded_shape[2:], [kx, ky], stride)]
    output_shape = [n, oc, oh, ow]
    output = np.zeros_like(data, shape=output_shape)
    for nn in range(n):
        for cc in range(oc):
            for p in range(oh):
                for q in range(ow):
                    sx, sy = stride
                    data_slice = padded_data[nn, :, p * sx: p * sx + kx, q * sy: q * sy + ky]
                    weight_slice = weight[cc, :, : data_slice.shape[1], :data_slice.shape[2]]
                    output[nn, cc, p, q] = np.sum(data_slice * weight_slice)
    return output


@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky, padding, stride",
    [
        [1, 3, 32, 32, 12, 3, 3, [1, 1, 1, 1], [1, 1]],  # kernel 3, stride 1
        [2, 3, 32, 32, 12, 3, 3, [1, 1, 1, 1], [1, 1]],  # kernel 3, stride 1, batch size 2
        [1, 3, 32, 32, 12, 3, 3, [0, 0, 0, 0], [2, 2]],  # kernel 3, stride 2
        [1, 3, 32, 32, 12, 1, 1, [1, 1, 1, 1], [1, 1]],  # kernel 1, stride 1
        [1, 3, 32, 32, 12, 1, 1, [0, 0, 0, 0], [2, 2]],  # kernel 1, stride 2
        [1, 3, 32, 32, 12, 7, 7, [3, 3, 3, 3], [2, 2]]  # kernel 7, stride 2
    ]
)
def test_conv2d(n, c, h, w, oc, kx, ky, padding, stride):
    check_binary(a_shape=[n, c, h, w], b_shape=[oc, c, kx, ky],
                 numpy_op=lambda data, weight: numpy_conv2d(data, weight, padding, stride),
                 hidet_op=lambda data, weight: ops.conv2d(data, weight, padding, stride),
                 dtype='float32', atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
