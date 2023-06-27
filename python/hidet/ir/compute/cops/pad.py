from typing import List
from hidet.ir.expr import if_then_else, logical_and, convert
from hidet.ir.compute.primitives import TensorNode, compute


def pad(data: TensorNode, pads: List[int], value: float):
    shape = data.shape
    rank = len(shape)
    assert rank * 2 == len(pads)
    out_shape = [a + b + c for a, b, c in zip(pads[:rank], shape, pads[rank:])]

    value = convert(value, dtype=data.type.dtype.name)

    def fmap(*indices):
        indices = [idx - beg for idx, beg in zip(indices, pads[:rank])]
        cond = logical_and(*[logical_and(0 <= idx, idx < shape[i]) for i, idx in enumerate(indices)])
        return if_then_else(cond, data[indices], value)

    out = compute('out', shape=out_shape, fcompute=fmap)
    return out
