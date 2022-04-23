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


def decompose(total, bases):
    ret = []
    cnt = 1
    for b in reversed(bases):
        ret.append(total % b)
        total = total // b
    ret.append(total)
    return list(reversed(ret))


def winograd_2d(input: np.ndarray, weight: np.ndarray):
    """ F(2, 3) in https://arxiv.org/pdf/1509.09308.pdf """
    m, r = 2, 3
    N, C, H, W = input.shape
    K = weight.shape[0]
    out_H = H - r + 1
    out_W = W - r + 1
    P = N * ((out_H + m - 1) // m) * ((out_W + m - 1) // m)  # only works when m - r + 1 == 0
    alpha = m + r - 1

    # d
    d = np.zeros([C, P, alpha, alpha], dtype=input.dtype)
    for c in range(C):
        for p in range(P):
            n, y, x = decompose(p, [(out_H + m - 1) // m, (out_W + m - 1) // m])
            y *= m
            x *= m
            print(y, x)
            tile = input[n, c, y: y + alpha, x: x + alpha]
            d[c, p, :tile.shape[0], :tile.shape[1]] = tile

    # g
    g = weight

    # G & BT
    G = np.array(
        [[1, 0, 0],
         [1/2, 1/2, 1/2],
         [1/2, -1/2, 1/2],
         [0, 0, 1]]
    )
    BT = np.array(
        [[1, 0, -1, 0],
         [0, 1, 1, 0],
         [0, -1, 1, 0],
         [0, 1, 0, -1]]
    )
    AT = np.array(
        [[1, 1, 1, 0],
         [0, 1, -1, -1]]
    )

    # u (K, C, alpha, alpha)
    U = np.expand_dims(G, (0, 1))  @ g @ np.expand_dims(G.transpose(), (0, 1))

    # v (C, P, alpha, alpha)
    V = np.expand_dims(BT, (0, 1)) @ d @ np.expand_dims(BT.transpose(), (0, 1))

    # m (alpha, alpha, K, P)
    M = np.transpose(U, (2, 3, 0, 1)) @ np.transpose(V, (2, 3, 0, 1))

    # y (K, P, m, m)
    Y = np.expand_dims(AT, (0, 1)) @ np.transpose(M, (2, 3, 0, 1)) @ np.expand_dims(AT.transpose(), (0, 1))

    # y (N, K, H', W')
    HH = (out_H + m - 1) // m * m
    WW = (out_W + m - 1) // m * m
    Y = np.reshape(Y, (K, N, (out_H + m - 1) // m, (out_W + m - 1) // m, m, m)).transpose((1, 0, 2, 4, 3, 5)).reshape((N, K, HH, WW))

    return Y[:N, :K, :out_H, :out_W]


if __name__ == '__main__':
    x = np.array([[[[1, 1, 1, 1, 2],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]]]], dtype=np.float32)
    w = np.array([[[[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]]]], dtype=np.float32)
    # x = np.random.randn(1, 3, 224, 224)
    # w = np.random.randn(64, 3, 3, 3)
    y = winograd_2d(x, w)
    print(y.shape)
    print(y)

'''
X: n c h w 
W: oc, c, r, r
ts = m + r - 1 (tile size)

oh = h - r + 1 (output height)
ow = w - r + 1 (output width)
nh = (oh + m - 1) / m (height tiles)
nw = (ow + m - 1) / m (width tiles)
p = n * nh * nw (number of tiles per input channel)

m, r

X => [c, n * nh * nw, ts, ts]
W => [oc, c, r, r]

U => [c, n * nh * nw, ts, ts]
V => [oc, c, ts, ts]

M => [oc, n * nh * nw, ts, ts]

Y => [oc, n * nh * nw, m, m] => [n, oc, oh, ow]
'''




