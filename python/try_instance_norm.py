import hidet
import torch
from hidet.graph.ops.normalize import instance_norm
import numpy as np
from hidet.graph.tensor import asarray

device = "cpu"
shapes = [[1, 32, 48], [1, 20, 20, 20], [1, 20, 20, 5, 5], [1, 32, 26214]]
shapes.extend([[10, 3, 3, 3, 4]])

def numpy_instance_norm(data: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    dims = tuple(range(2, len(data.shape)))
    mean = data.mean(axis=dims, keepdims=True)
    var = data.var(axis=dims, keepdims=True)
    return (data - mean) / np.sqrt(var + epsilon)
dtype = "float32"
for shape in shapes:
    data = np.random.randn(*shape).astype(dtype)
    a = asarray(data).to(device=device)
    a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype='float32'))
    m = torch.nn.functional.instance_norm(a_torch)
    # m = numpy_instance_norm(data)
    # print(np.allclose(np_layernorm(np.array(a.numpy(), copy=True, dtype='float32')), m(a_torch).detach().numpy()))
    xx = hidet.symbol(shape, dtype="float32", device=device)
    yy = instance_norm(xx, epsilon=1e-5)
    op: hidet.Operator = yy.op
    compiled_func = op.compiled_task.candidates[0]
    b = hidet.zeros(shape, device=device)
    compiled_func(a, b)
    np.testing.assert_allclose(b.numpy(), m, rtol=1e-4, atol=1e-4)
    hidet_latency = hidet.utils.benchmark_func(lambda: compiled_func(a, b), warmup=10, repeat=50)
    pt_latency = hidet.utils.benchmark_func(lambda: torch.nn.functional.instance_norm(a_torch), warmup=10, repeat=50)
    print("shape", shape,   "hidet:", hidet_latency, "pytorch:", pt_latency)
    print("fastest is:", ["hidet", "pytorch"][np.argmin([hidet_latency, pt_latency])], "\n")
    print("hidet output tensor is correct")
