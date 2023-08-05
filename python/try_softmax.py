import numpy as np
import torch
# torch.nn.functional.softmax()
import hidet
from hidet.graph.ops import softmax
import torch.nn as nn
# shapes = [([1, 2, 3], 1), ([8, 8, 8, 8], 0), ([8, 8, 8, 8], 1), ([8, 8, 8, 8], 2), ([8, 8, 8, 8], 3), ([2, 2, 8], 0),
#           ([1, 2, 16], 1), ([8, 8, 8], 1), ([8, 1000], -1), ([32, 512, 512], -1), ([32, 512, 512], 1),
#           ([8, 3, 224, 224], -1), ([32, 128, 768], 1)]
shapes = [
    ([6, 6], 0),
    ([5, 5, 5], 1),
    ([2, 2, 2, 2, 2, 2], 3)
]
shapes = [
    # ([10, 20, 40, 30, 50], 2),
    # ([5, 5, 80, 100, 70], 1),
    # ([8, 60, 90, 100, 35], 0),
    ([12, 8, 7, 43], 2),
    # ([9, 24, 36, 55], 1),
    # ([7, 19, 27, 38], 0),
    # ([21, 34, 22, 77], 1),
    ([16, 28, 30, 44], 2),
]

# shapes = [([4, 100], -1)]
hidet.option.search_space(0)
# hidet.option.runtime_check(False)
for shape, axis in shapes:
    a = hidet.randn(shape, device="cpu")
    xx = hidet.symbol(shape, dtype="float32", device="cpu")
    yy = softmax(xx, axis=axis)
    op: hidet.Operator = yy.op
    compiled_func = op.compiled_task.candidates[0]
    b = hidet.zeros(shape, device="cpu")

    compiled_func(a, b)
    device = torch.device("cpu")
    m = nn.Softmax(dim=axis)
    a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype='float32'))
    # print(a)
    # print(b, m(a_torch))
    np.testing.assert_allclose(b.numpy(), m(a_torch), rtol=1e-05, atol=1e-08)
    print("hidet and pytorch tensors match")

    def numpy_softmax(data, axis_):
        data = np.exp(data - np.max(data, axis_, keepdims=True))
        data = data / np.sum(data, axis_, keepdims=True)
        return data

    # hidet_latency = hidet.utils.benchmark_func(lambda: compiled_func(a, b), warmup=10, repeat=50)
    # pt_latency = hidet.utils.benchmark_func(lambda: m(a_torch), warmup=10, repeat=50)
    # np_latency = hidet.utils.benchmark_func(lambda: numpy_softmax(a.numpy(), axis_=axis), warmup=10, repeat=50)
    # print("shape", shape, "and axis", axis, "hidet:", hidet_latency, "pytorch:", pt_latency, "numpy:", np_latency)
    # print("fastest is:", ["hidet", "pytorch", "numpy"][np.argmin([hidet_latency, pt_latency, np_latency])])
    # print(b, m(a_torch))
# softmax([bs, 1000], axis=1)  # bs = 1, 2, 4, 8
# softmax([heads, seq, seq], axis=2)  # heads=32, seq = 128, 512, 1024

