import numpy as np
import torch

import hidet
from hidet.graph.ops.definitions.activation import softmax
from scipy.special import softmax as softmax_scipy
import timeit
import time
import torch.nn as nn

a = hidet.randn([8, 8], device="cpu")
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

np_latency = hidet.utils.benchmark_func(
    lambda: np.max(a.numpy(), axis=1), repeat=30
)

print(hidet_latency, np_latency)


b = np.zeros((8, 8))
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        b[i,j] = float(a[i,j])

def sftmx(a):
    b = np.zeros_like(a)
    for i in range(a.shape[0]):
        c = np.exp(a[i])
        b[i] = c/np.sum(c)
    return b


print(a_np.dtype)
print(np.allclose(softmax(a), sftmx(b)))
