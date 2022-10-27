from typing import List, Union
from ..utils import normalize_stride


def infer_conv2d_shape(
    x_shape: List[int], w_shape: List[int], strides: Union[int, List[int]], groups: int
) -> List[int]:
    n, c, h, w = x_shape
    oc, gc, kx, ky = w_shape
    sx, sy = normalize_stride(strides)
    if gc * groups != c:
        msg = 'Conv2d: x has {} input channels, w has {} group channels, and groups={}'.format(c, gc, groups)
        raise ValueError(msg)
    if oc % groups != 0:
        msg = 'Conv2d expects out_channels % groups == 0, got out_channels {} and groups {}'.format(oc, groups)
        raise ValueError(msg)
    p, q = (h - kx) // sx + 1, (w - ky) // sy + 1
    return [n, oc, p, q]
