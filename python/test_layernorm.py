import torch
from hidet.graph.ops.normalize import layer_norm
import hidet
import numpy as np

shape = [1, 2, 8, 9]
dims = 2
a = hidet.randn(shape, device="cuda")
x1 = hidet.symbol_like(a)
y = layer_norm(x1, num_last_dims=dims, epsilon=1e-5)

graph: hidet.FlowGraph = hidet.trace_from(y, inputs=[x1])
opt_graph = hidet.graph.optimize(graph)
# compiled_func = opt_graph.nodes[0].compiled_task.candidates[0]
# b = hidet.zeros(shape, device="cuda")

b = opt_graph(a)  # opt graph for correct output, compiledmodule for fast? weird asf lol
print(hidet.option.get_cache_dir())
b = layer_norm(a, num_last_dims=dims)  # this works but flowgraph doesn't?
# Also, running using the compiledmodule as above doesn't do any codegen in .cache/hidet

m = torch.nn.LayerNorm(shape[-dims:], eps=1e-5)
a = a.to(device="cpu")
a_torch = torch.from_numpy(np.array(a.numpy(), copy=True, dtype='float32'))
print(np.allclose(b.to(device="cpu").numpy(), m(a_torch).detach().numpy()))
