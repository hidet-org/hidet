from hidet.tos.graph import FlowGraph
from .fold_const import fold_const_pass
from .pattern_transform import pattern_transform_pass

# [x] fold_const
# [ ] fuse_affine
# [ ] fuse_conv
# [ ] fuse_matmul
# [ ] fuse_elementwise


def optimize(graph: FlowGraph) -> FlowGraph:
    passes = [
        fold_const_pass(),
        pattern_transform_pass()
    ]
    for p in passes:
        graph = p(graph)
    return graph.update_nodes()
