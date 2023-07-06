import numpy as np
import torch
# torch.nn.functional.softmax()
import hidet
from hidet.graph.ops import softmax
import torch.nn as nn
shape = [50, 1005]
# hidet.option.search_space(0)
# hidet.option.runtime_check(False)
a = hidet.randn(shape, device="cpu")
# a = hidet.randn([2, 8, 8], device="cpu")
print(a)
# print(timeit.timeit('softmax(a)',
#                     setup='from __main__ import softmax, a'))
# print(timeit.timeit('np.max(a_np, axis=1)',
#                     setup='from __main__ import a_np, np'))
# start_time = time.time()
x1 = hidet.symbol_like(a)
y = softmax(x1)

graph: hidet.FlowGraph = hidet.trace_from(y, inputs=[x1])
opt_graph = hidet.graph.optimize(graph)
compiled_func = opt_graph.nodes[0].compiled_task.task_module
b = hidet.zeros(shape, device="cpu")

compiled_func(a, b)

device = torch.device("cpu")
m = nn.Softmax(dim=1)
a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype=float))
print(np.allclose(b.numpy(), m(a_torch)))

hidet_latency = hidet.utils.benchmark_func(
    lambda: compiled_func(a, b), warmup=10, repeat=50
)
np_latency = hidet.utils.benchmark_func(
    lambda: m(a_torch), warmup=10, repeat=50
)
# print(compiled_func.profile(a, b))
print(hidet_latency, np_latency)
# print(b)
# print(m(a_torch))

# softmax([bs, 1000], axis=1)  # bs = 1, 2, 4, 8
# softmax([heads, seq, seq], axis=2)  # heads=32, seq = 128, 512, 1024

