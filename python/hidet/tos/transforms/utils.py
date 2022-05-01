from hidet.tos.ir import Operator


def is_barrier(op: Operator):
    from hidet.tos.ops.definitions.special import BarrierOp
    return isinstance(op, BarrierOp)
