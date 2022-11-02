from typing import List
from ..expr import Expr, convert


def index_serialize(indices: List[Expr], shape: List[int]) -> Expr:
    if len(shape) == 0:
        return convert(0)
    scalar_index: Expr = convert(0)
    acc = 1
    for idx_value, extent in reversed(list(zip(indices, shape))):
        scalar_index += idx_value * acc
        acc *= extent
    return scalar_index


def index_deserialize(scalar_index: Expr, shape: List[int]) -> List[Expr]:
    if len(shape) == 0:
        return []
    indices = []
    acc = 1
    for r, extent in enumerate(reversed(shape)):
        if r < len(shape) - 1:
            indices.append(scalar_index // acc % extent)
        else:
            indices.append(scalar_index // acc)
        acc *= extent
    return list(reversed(indices))
