import numpy as np
import torch

import hidet
from hidet.graph.ops.definitions.activation import softmax
from scipy.special import softmax as softmax_scipy
import timeit
import time
import torch.nn as nn

a = hidet.randn([1, 2000], device="cpu")
# a = hidet.randn([2, 8, 8], device="cpu")
print(a)
# print(timeit.timeit('softmax(a)',
#                     setup='from __main__ import softmax, a'))
a_np = a.numpy()
# print(timeit.timeit('np.max(a_np, axis=1)',
#                     setup='from __main__ import a_np, np'))
# start_time = time.time()
b = softmax(a)
# print(time.time() - start_time)
#
# start_time = time.time()
# b = np.max(a_np, axis=1)
# print(time.time() - start_time)

print(b)

hidet_latency = hidet.utils.benchmark_func(
    lambda: softmax(a), repeat=30
)
# def manualmax(a):
#     row_size = 8
#     col_size = 8
#     for i in range(row_size):
#         max_val = a[i, 0]
#         for j in range(col_size):
#             max_val = a[i, j] if max_val < a[i, j] else max_val
#         a[i, 0] = max_val
device = torch.device("cpu")
m = nn.Softmax(dim=1)
a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype=float))
np_latency = hidet.utils.benchmark_func(
    lambda: m(a_torch), repeat=30
)
print(m(a_torch))
print(hidet_latency, np_latency)

# softmax([bs, 1000], axis=1)  # bs = 1, 2, 4, 8
# softmax([heads, seq, seq], axis=2)  # heads=32, seq = 128, 512, 1024

# TODO: spend a lot of time looking at pytorch's c++ implementations of softmax
# TODO: and make sure that the pytorch ones are actually used and not a reference implementation!


# b = np.zeros((8, 8))
# for i in range(a.shape[0]):
#     for j in range(a.shape[1]):
#         b[i,j] = float(a[i,j])

# def sftmx(a):
#     b = np.zeros_like(a)
#     for i in range(a.shape[0]):
#         c = np.exp(a[i])
#         b[i] = c/np.sum(c)
#     return b


print(a_np.dtype)
print(np.allclose(softmax(a).numpy(), m(a_torch)))
