import sys

import numpy as np
import torch
# torch.nn.functional.softmax()
import hidet
from hidet.graph.ops import softmax
import torch.nn as nn
shapes = []
shapes.extend([([1, 2, 3], 1), ([8, 8, 8, 8], 0), ([8, 8, 8, 8], 1), ([8, 8, 8, 8], 2), ([8, 8, 8, 8], 3),
               ([2, 2, 8], 0), ([1, 2, 16], 1), ([8, 8, 8], 1), ([8, 1000], -1), ([32, 512, 512], -1),
               ([32, 512, 512], 1), ([8, 3, 224, 224], -1), ([32, 128, 768], 1)])
shapes.extend([
    ([6, 6], 0),
    ([5, 5, 5], 1),
    ([2, 2, 2, 2, 2, 2], 3)
])
shapes.extend([
    ([12, 8, 7, 43], 2),
    ([2, 1, 9], 0),
    ([2, 2, 2, 9], 1),
    ([1, 2, 9], 0),
    ([2, 2, 9], 0),
    ([9, 24, 36, 55], 1),
    ([7, 19, 27, 38], 0),
    ([21, 34, 22, 77], 1),
    ([16, 28, 30, 44], 2),
])
# shapes=[([32, 512, 512], 1)]

# shapes = [([4, 100], -1)]
shapes = [([1, 1000], 1), ([16, 1000], 1), ([16, 1000, 1, 1], -1), ([1, 128, 128, 128], 2)]
hidet.option.search_space(0)
shapes = [([1, ("x", 1000), ('y', 1), 1], 1), ([1, ("x", 1000)], 1), ([("x", 16), 1000], 1),
          ([("x", 16), ("y", 1000), ("z", 1), ("w", 1)], 1), ([1, ("x", 128), ("y", 128), ("z", 128)], 2)]
# hidet.option.runtime_check(False)
hidetvspt = []
for shape, axis in shapes:
    shapec = shape
    shape = [(i if isinstance(i, int) else i[0]) for i in shape]
    concrete_shape = [(i if isinstance(i, int) else i[1]) for i in shapec]
    dtype = "float32"
    device = "cpu"
    from hidet.graph.tensor import asarray
    data = np.array(np.random.randn(*concrete_shape)).astype(dtype)
    hidet_data = asarray(data).to(device=device)
    m = nn.Softmax(dim=axis)
    res = m(torch.from_numpy(data))
    sym = hidet.symbol(shape, dtype=dtype, device=device)
    out = softmax(sym)
    func = hidet.trace_from(out, sym).build()
    hidet_res = func(hidet_data).numpy()
    np.testing.assert_allclose(actual=hidet_res, desired=res, atol=1e-8, rtol=1e-5)
    print("here")

    # a = hidet.randn(shape, device="cpu")
    # xx = hidet.symbol(shape, dtype="float32", device="cpu")
    # yy = softmax(xx, axis=axis)
    # op: hidet.Operator = yy.op
    # compiled_func = op.compiled_task.candidates[0]
    # b = hidet.zeros(shape, device="cpu")
    #
    # compiled_func(a, b)
    # device = torch.device("cpu")
    # m = nn.Softmax(dim=axis)
    # a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype='float32'))
    # # print(a)
    # # print(b, m(a_torch))
    # np.testing.assert_allclose(b.numpy(), m(a_torch), rtol=1e-05, atol=1e-08)
    # print("hidet and pytorch tensors match")
    #
    # def numpy_softmax(data, axis_):
    #     data = np.exp(data - np.max(data, axis_, keepdims=True))
    #     data = data / np.sum(data, axis_, keepdims=True)
    #     return data

    hidet_latency = hidet.utils.benchmark_func(lambda: func(hidet_data), warmup=10, repeat=50)
    pt_latency = hidet.utils.benchmark_func(lambda: m(torch.from_numpy(data)), warmup=10, repeat=50)
    print("shape", shape, "and axis", axis, "hidet:", hidet_latency, "pytorch:", pt_latency)
    print("fastest is:", ["hidet", "pytorch"][np.argmin([hidet_latency, pt_latency])], "\n")
    hidetvspt.append((shape, axis if axis >= 0 else len(shape) + axis, pt_latency/hidet_latency))
    # print(b, m(a_torch))
for shape, axis, speed in hidetvspt:
    print("shape:", shape, "axis:", axis, "hidet vs pt speed:", speed)
# softmax([bs, 1000], axis=1)  # bs = 1, 2, 4, 8
# softmax([heads, seq, seq], axis=2)  # heads=32, seq = 128, 512, 1024

