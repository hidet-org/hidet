import numpy as np

from hidet import nn
import hidet
import torch
from hidet.graph.ops.normalize import layer_norm


shapes = [[2, 2, 30, 30]]
for shape in shapes:
    a = hidet.randn(shape, device="cpu")
    print(a.dtype)
    x1 = hidet.symbol_like(a)
    y = layer_norm(x1, num_last_dims=1, epsilon=1e-5)

    graph: hidet.FlowGraph = hidet.trace_from(y, inputs=[x1])
    opt_graph = hidet.graph.optimize(graph)
    compiled_func = opt_graph.nodes[0].compiled_task.candidates[0]
    b = hidet.zeros(shape, device="cpu")

    compiled_func(a, b)
    # b = y(a)

    a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype='float32'))
    m = torch.nn.LayerNorm(shape[-1:], eps=1e-5)
    print(b, m(a_torch))
    print(np.allclose(b.numpy(), m(a_torch).detach().numpy(), atol=1e-7))  # erm default abs tolerance doesnt work

