from hidet.graph.ir import FlowGraph

from .base import GraphPass, PassContext, logger
from .instruments import GraphPassInstrument, SaveGraphInstrument, ProfileInstrument
from .fold_const import fold_const_pass
from .pattern_transform import pattern_transform_pass
from .automatic_mix_precision import automatic_mix_precision_pass
from .resolve_variant import resolve_variant_pass
from .fuse_operator import fuse_operator_pass
from .eliminate_barrier import eliminate_barrier_pass

from .resolve_variant import ResolveRule, register_resolve_rule, get_resolve_chain


def optimize(graph: FlowGraph) -> FlowGraph:
    """Optimize a flow graph.

    This function applies a sequence of predefined graph-level passes to a :class:`~hidet.graph.FlowGraph` to
    conduct optimizations and graph transformations.

    .. tip::

        Some graph passes provide options to config, please refer to :class:`hidet.graph.PassContext` for more
        information on graph pass configuration.

    Parameters
    ----------
    graph: FlowGraph
        The flow graph to be optimized.

    Returns
    -------
    ret: FlowGraph
        The optimized flow graph.
    """
    passes = [
        fold_const_pass(),
        pattern_transform_pass(),
        automatic_mix_precision_pass(),
        resolve_variant_pass(),
        fuse_operator_pass(),
        eliminate_barrier_pass(),
    ]
    ctx = PassContext.current()
    for inst in ctx.instruments:
        inst.before_all_passes(graph)
    for optimize_pass in passes:
        graph = optimize_pass(graph)
    for inst in reversed(ctx.instruments):
        inst.after_all_passes(graph)
    return graph.update_nodes()
