from hidet.tos.ir.graph import FlowGraph
from .base import PassContext
from .fold_const import fold_const_pass
from .pattern_transform import pattern_transform_pass
from .fuse_elementwise import fuse_elementwise_pass


def optimize(graph: FlowGraph) -> FlowGraph:
    passes = [
        fold_const_pass(),
        pattern_transform_pass(),
        # fuse_elementwise_pass()
    ]
    ctx = PassContext.current()
    for inst in ctx.instruments:
        inst.before_all_passes(graph)
    for p in passes:
        graph = p(graph)
    for inst in reversed(ctx.instruments):
        inst.after_all_passes(graph)
    return graph.update_nodes()
