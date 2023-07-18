import numpy as np
import torch

import hidet
from hidet.graph.ops.definitions.activation import softmax
from scipy.special import softmax as softmax_scipy
import timeit
import time
import torch.nn as nn
shape = [4, 1000]
a = hidet.randn(shape, device="cpu")
# a = hidet.randn([2, 8, 8], device="cpu")
print(a)
# print(timeit.timeit('softmax(a)',
#                     setup='from __main__ import softmax, a'))
a_np = a.numpy()
# print(timeit.timeit('np.max(a_np, axis=1)',
#                     setup='from __main__ import a_np, np'))
# start_time = time.time()
x1 = hidet.symbol_like(a)
y = softmax(x1)

graph: hidet.FlowGraph = hidet.trace_from(y, inputs=[x1])
opt_graph = hidet.graph.optimize(graph)
compiled_func = opt_graph.nodes[0].task_func
b = hidet.zeros(shape, device="cpu")

compiled_func(a, b)

print(b)

hidet_latency = hidet.utils.benchmark_func(
    lambda: compiled_func(a, b), repeat=50
)
device = torch.device("cpu")
m = nn.Softmax(dim=1)
a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype=float))
np_latency = hidet.utils.benchmark_func(
    lambda: m(a_torch), repeat=50
)
print(m(a_torch))
print(hidet_latency, np_latency)

# softmax([bs, 1000], axis=1)  # bs = 1, 2, 4, 8
# softmax([heads, seq, seq], axis=2)  # heads=32, seq = 128, 512, 1024

# TODO: spend a lot of time looking at pytorch's c++ implementations of softmax
# TODO: and make sure that the pytorch ones are actually used and not a reference implementation!


print(a_np.dtype)
print(np.allclose(softmax(a).numpy(), m(a_torch)))
