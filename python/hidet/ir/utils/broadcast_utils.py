from typing import Sequence, List
from hidet.ir.expr import Expr, Int, is_constant, if_then_else


def can_broadcast(src_shape: Sequence[Int], dst_shape: Sequence[Int]) -> bool:
    if len(dst_shape) < len(src_shape):
        return False
    src_shape = [1 for _ in range(len(dst_shape) - len(src_shape))] + list(src_shape)
    for a, b in zip(src_shape, dst_shape):
        if is_constant(a, b) and a not in [1, b]:
            return False
    return True


def can_mutually_broadcast(x_shape: Sequence[Int], y_shape: Sequence[Int]) -> bool:
    x_shape, y_shape = list(x_shape), list(y_shape)
    while len(x_shape) < len(y_shape):
        x_shape = [1] + x_shape
    while len(y_shape) < len(x_shape):
        y_shape = [1] + y_shape
    return all(p == q or p == 1 or q == 1 for p, q in zip(x_shape, y_shape) if is_constant(p, q))


def broadcast_shape(x_shape: Sequence[Int], y_shape: Sequence[Int]) -> List[Int]:
    """
    Broadcast two shapes with the same rule as numpy.
    Please refer to https://numpy.org/doc/stable/user/basics.broadcasting.html for details.
    """
    from hidet.ir.dtypes import int32

    x_shape, y_shape = list(x_shape), list(y_shape)
    orig_shapes = x_shape, y_shape
    while len(x_shape) < len(y_shape):
        x_shape = [int32(1)] + x_shape
    while len(y_shape) < len(x_shape):
        y_shape = [int32(1)] + y_shape
    result_shape = []
    for p, q in zip(x_shape, y_shape):
        if is_constant(p) and p == 1:
            result_shape.append(q)
        elif is_constant(q) and q == 1:
            result_shape.append(p)
        elif is_constant(p, q):
            if p != q:
                raise ValueError(
                    'can not broadcast two arrays with shape {} and {}'.format(orig_shapes[0], orig_shapes[1])
                )
            result_shape.append(p)
        else:
            result_shape.append(p if is_constant(p) else q)
    return result_shape


def broadcast_shapes(shapes: Sequence[Sequence[Int]]) -> List[Int]:
    assert len(shapes) >= 1
    expanded_shape = list(shapes[0])
    for shape in shapes:
        expanded_shape = broadcast_shape(expanded_shape, shape)
    return expanded_shape


def broadcast_indices(out_indices: Sequence[Int], shape: Sequence[Int], out_shape: Sequence[Int]) -> List[Expr]:
    if len(out_indices) != len(out_shape):
        raise ValueError('Number of indices {} does not match the output shape {}'.format(out_indices, out_shape))

    pad_dim = len(out_shape) - len(shape)
    indices = list(out_indices[pad_dim:])
    for idx, dim in enumerate(shape):
        indices[idx] = if_then_else(dim == 1, 0, indices[idx])
    return indices
