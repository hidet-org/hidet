from hidet.tos.ir import FlowGraph

from .base import GraphPass, PassContext, logger
from .instruments import GraphPassInstrument, SaveGraphInstrument, ProfileInstrument
from .fold_const import fold_const_pass
from .pattern_transform import pattern_transform_pass
from .automatic_mix_precision import automatic_mix_precision_pass
from .resolve_mma import resolve_mma_pass
from .resolve_variant import resolve_variant_pass
from .fuse_unary_elementwise import fuse_unary_elementwise_pass
from .fuse_epilogue import fuse_epilogue_pass
from .fuse_prologue import fuse_prologue_pass
from .eliminate_barrier import eliminate_barrier_pass


def optimize(graph: FlowGraph) -> FlowGraph:
    passes = [
        fold_const_pass(),
        pattern_transform_pass(),
        automatic_mix_precision_pass(),
        resolve_variant_pass(),
        resolve_mma_pass(),
        fuse_unary_elementwise_pass(),
        fuse_epilogue_pass(),
        fuse_prologue_pass(),
        eliminate_barrier_pass()
    ]
    ctx = PassContext.current()
    for inst in ctx.instruments:
        inst.before_all_passes(graph)
    for optimize_pass in passes:
        graph = optimize_pass(graph)
    for inst in reversed(ctx.instruments):
        inst.after_all_passes(graph)
    return graph.update_nodes()
