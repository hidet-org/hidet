import hidet
import torch
from hidet.graph.ops.normalize import batch_norm_infer
import numpy as np
from hidet.graph.tensor import asarray

device = "cpu"
shapes = [[1, 1, 1, 1], [1, 200, 20, 20], [1, 10, 1, 1], [1, 128, 32, 32], [1, 32, 24, 24]]

dtype = "float32"
for shape in shapes:
    a = hidet.randn(shape, device=device)
    b = hidet.randn([shape[1]], device=device)
    c = hidet.randn([shape[1]], device=device)
    a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype='float32'))
    rmean = torch.from_numpy(np.array(b.numpy(), copy=True, dtype='float32'))
    rvar = torch.from_numpy(np.array(c.numpy(), copy=True, dtype='float32'))
    m = torch.nn.functional.batch_norm(a_torch, rmean, rvar)
    # m = numpy_instance_norm(data)
    # print(np.allclose(np_layernorm(np.array(a.numpy(), copy=True, dtype='float32')), m(a_torch).detach().numpy()))
    xx = hidet.symbol(shape, dtype="float32", device=device)
    xxx = hidet.symbol([shape[1]], dtype="float32", device=device)
    xxxx = hidet.symbol([shape[1]], dtype="float32", device=device)
    yy = batch_norm_infer(xx, xxx, xxxx, epsilon=1e-5)
    op: hidet.Operator = yy.op
    compiled_func = op.compiled_task.candidates[0]
    o = hidet.zeros(shape, device=device)
    compiled_func(a, b, c, o)
    np.testing.assert_allclose(o.numpy(), m, rtol=1e-4, atol=1e-4)
    hidet_latency = hidet.utils.benchmark_func(lambda: compiled_func(a, b, c, o), warmup=10, repeat=50)
    pt_latency = hidet.utils.benchmark_func(lambda: torch.nn.functional.instance_norm(a_torch, rmean, rvar), warmup=10, repeat=50)
    print("shape", shape, "hidet:", hidet_latency, "pytorch:", pt_latency)
    print("fastest is:", ["hidet", "pytorch"][np.argmin([hidet_latency, pt_latency])], "\n")
    print("hidet output tensor is correct")
