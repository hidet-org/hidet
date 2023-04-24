import hidet
hidet.option.cache_dir('.')

def fused_layer(x, y):
    a = x + y
    b = x * y
    c = hidet.ops.batch_matmul(a, b)
    o = c + a
    return o

x = hidet.symbol([1, 1024, 1024], dtype='float32', device='cuda')
y = hidet.symbol([1, 1024, 1024], dtype='float32', device='cuda')
o = fused_layer(x, y)
graph = hidet.trace_from(o,[x, y])
print(graph.latency())
# graph_opt = hidet.graph.optimize(graph)
# print(graph_opt.latency())