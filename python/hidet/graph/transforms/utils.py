from hidet.graph.ir import Operator


def is_barrier(op: Operator):
    from hidet.graph.ops.definitions.special import BarrierOp
    return isinstance(op, BarrierOp)
