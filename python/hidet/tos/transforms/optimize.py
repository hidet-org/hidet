from hidet.tos.graph import FlowGraph
from .fold_const import fold_const_pass

# fold_const
# fuse_affine
# fuse_conv
# fuse_matmul
# fuse_elementwise


def optimize(graph: FlowGraph) -> FlowGraph:
    passes = [
        fold_const_pass()
    ]
    for p in passes:
        graph = p(graph)
    return graph.finish()
