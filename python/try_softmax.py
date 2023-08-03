import numpy as np
import torch
# torch.nn.functional.softmax()
import hidet
from hidet.graph.ops import softmax
import torch.nn as nn
shapes = [([8, 8, 8], 1), ([8, 1000], -1), ([32, 512, 512], -1), ([32, 512, 512], 1), ([8, 3, 224, 224], -1),
          ([32, 128, 768], 1)]
# shapes = [([4, 100], -1)]
hidet.option.search_space(0)
# hidet.option.runtime_check(False)
for shape, axis in shapes:
    a = hidet.randn(shape, device="cpu")
    x1 = hidet.symbol_like(a)
    y = softmax(x1, axis=axis)

    graph: hidet.FlowGraph = hidet.trace_from(y, inputs=[x1])
    opt_graph = hidet.graph.optimize(graph)
    compiled_func = opt_graph.nodes[0].compiled_task.candidates[0]
    b = hidet.zeros(shape, device="cpu")

    compiled_func(a, b)

    device = torch.device("cpu")
    m = nn.Softmax(dim=axis)
    a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype='float32'))

    np.testing.assert_allclose(b.numpy(), m(a_torch), rtol=1e-05, atol=1e-08)
    print("hidet and pytorch tensors match")

    def numpy_softmax(data, axis_):
        data = np.exp(data - np.max(data, axis_, keepdims=True))
        data = data / np.sum(data, axis_, keepdims=True)
        return data

    hidet_latency = hidet.utils.benchmark_func(lambda: compiled_func(a, b), warmup=10, repeat=50)
    pt_latency = hidet.utils.benchmark_func(lambda: m(a_torch), warmup=10, repeat=50)
    np_latency = hidet.utils.benchmark_func(lambda: numpy_softmax(a.numpy(), axis_=axis), warmup=10, repeat=50)
    print("for shape of", shape, ":", "hidet:", hidet_latency, "pytorch:", pt_latency, "numpy:", np_latency)
    print("fastest is:", ["hidet", "pytorch", "numpy"][np.argmin([hidet_latency, pt_latency, np_latency])])
    # print(b, m(a_torch))
# softmax([bs, 1000], axis=1)  # bs = 1, 2, 4, 8
# softmax([heads, seq, seq], axis=2)  # heads=32, seq = 128, 512, 1024

