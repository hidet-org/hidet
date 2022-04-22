from typing import List
import numpy as np


def winograd_1d_tile(s: np.ndarray, w: np.ndarray):
    n = s.shape[0]  # input size
    r = w.shape[0]  # filter size
    assert n == 4 and r == 3
    s = np.array([s[0] - s[2], s[1] + s[2], s[2] - s[1], s[1] - s[3]])
    w = np.array([w[0], (w[0] + w[1] + w[2]) / 2, (w[0] - w[1] + w[2]) / 2, w[2]])
    c = s * w
    out = np.array([c[0] + c[1] + c[2], c[1] - c[2] - c[3]])
    return out


def winograd_1d(seq: np.ndarray, weight: np.ndarray):
    """ F(2, 3) in https://arxiv.org/pdf/1509.09308.pdf """
    n = seq.shape[0]  # input size
    r = weight.shape[0]  # filter size

    # input transform
    m = n - r + 1  # output size


if __name__ == '__main__':
    out = winograd_1d_tile(
        np.array([1, 2, 3, 4]),
        np.array([1, 1, 1])
    )
    print(out)

