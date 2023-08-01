import numpy as np

from hidet import nn
import hidet
import torch
from hidet.graph.ops.normalize import layer_norm
torch.set_printoptions(8)

d = 1
shapes = [([1, 2, 8, 8], d), ([2, 2, 2, 255], d), ([1, 8], 1), ([1, 1, 1, 18], d), ([2, 2, 45, 45], d),
          ([2, 2, 1, 1], d), ([512, 768], 1)]
for i, (shape, num_last_dims) in enumerate(shapes):
    a = hidet.randn(shape, device="cpu")
    x1 = hidet.symbol_like(a)
    y = layer_norm(x1, num_last_dims=num_last_dims, epsilon=1e-5)

    graph: hidet.FlowGraph = hidet.trace_from(y, inputs=[x1])
    opt_graph = hidet.graph.optimize(graph)
    compiled_func = opt_graph.nodes[0].compiled_task.candidates[0]
    b = hidet.zeros(shape, device="cpu")

    compiled_func(a, b)
    # b = y(a)
    # a = a.to(device="cpu")
    # b = b.to(device="cpu")
    a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype='float32'))
    # TODO: torch inaccuracy because it uses bfloat16 and not f32? not sure here but cant test on f64
    m = torch.nn.LayerNorm(shape[-num_last_dims:], eps=1e-5)
    # if i == 2:
    #     print(b, m(a_torch))
    print(shape)
    # print(b)
    atol = 0.001
    a_cuda = a.to(device="cuda")
    b_cuda = layer_norm(a_cuda, num_last_dims=num_last_dims)
    print(b_cuda)
    print(np.allclose(b.numpy(), b_cuda.to(device="cpu").numpy(), atol=atol))
    correct = np.allclose(b.numpy(), m(a_torch).detach().numpy(), atol=atol)  # default abs tol doesnt work cuz avxrsqrt
    hidet_latency = hidet.utils.benchmark_func(lambda: compiled_func(a, b), warmup=10, repeat=50)
    pt_latency = hidet.utils.benchmark_func(lambda: m(a_torch), warmup=10, repeat=50)
    print("for shape of", shape, "with num_last_dims =", num_last_dims, ":",
          "hidet:", hidet_latency, "pytorch:", pt_latency)
    print("fastest is:", ["hidet", "pytorch"][np.argmin([hidet_latency, pt_latency])])
    assert correct, "HIDET AND PYTORCH OUTPUTS WRONG FOR TOLERANCE " + str(atol)
    print("hidet and pytorch outputs match")

    # inaccuracy due to _mm256_rsqrt_ps having max error of 1.5x2^-12 which is kinda high
