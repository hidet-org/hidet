from typing import Sequence
from hidet.ir.expr import Expr, Var, is_constant
from hidet.ir.compute.primitives import TensorNode, compute, reduce
from hidet.ir.utils import broadcast_shape, broadcast_indices


def is_true(cond: Expr) -> bool:
    if is_constant(cond):
        return bool(cond) is True
    return False


def is_false(cond: Expr) -> bool:
    if is_constant(cond):
        return bool(cond) is False
    return False


def matmul(a: TensorNode, b: TensorNode, allow_1d=False, ta=False, tb=False) -> TensorNode:
    # The semantics of this operator is the same as the one in numpy
    # See Also https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    if len(a.shape) <= 1 and len(b.shape) <= 1:
        raise ValueError('At least one of the inputs must have rank > 1')
    if (len(a.shape) < 2 or len(b.shape) < 2) and not allow_1d:
        raise ValueError('Both inputs must have rank >= 2')
    if len(a.shape) == 1:
        if is_true(a.shape[0] != b.shape[-2]):
            raise ValueError('Cannot multiply matrices with shape {} and {}.'.format(a.shape, b.shape))
        reduce_extent = a.shape[0]
        if not tb:
            c_shape = b.shape[:-2] + [b.shape[-2]]
        else:
            c_shape = b.shape[:-2] + [b.shape[-1]]
    elif len(b.shape) == 1:
        if is_true(a.shape[-1] != b.shape[0]):
            raise ValueError('Cannot multiply matrices with shape {} and {}.'.format(a.shape, b.shape))
        reduce_extent = a.shape[-1]
        if not ta:
            c_shape = a.shape[:-2] + [a.shape[-2]]
        else:
            c_shape = a.shape[:-2] + [a.shape[-1]]
    else:
        if not ta:
            a_m_size, a_k_size = a.shape[-2], a.shape[-1]
        else:
            a_m_size, a_k_size = a.shape[-1], a.shape[-2]
        if not tb:
            b_k_size, b_n_size = b.shape[-2], b.shape[-1]
        else:
            b_k_size, b_n_size = b.shape[-1], b.shape[-2]

        if is_true(a_k_size != b_k_size):
            raise ValueError('Cannot multiply matrices with shape {} and {}.'.format(a.shape, b.shape))
        reduce_extent = a_k_size
        c_shape = broadcast_shape(a.shape[:-2], b.shape[:-2]) + [a_m_size, b_n_size]

    def fcompute(indices: Sequence[Var], k: Var) -> Expr:
        indices = list(indices)
        if len(a.shape) == 1:
            a_val = a[k]
            if not tb:
                b_val = b[indices[:-1] + [k] + indices[-1:]]
            else:
                b_val = b[indices[:-1] + indices[-1:] + [k]]
        elif len(b.shape) == 1:
            if not ta:
                a_val = a[indices + [k]]
            else:
                a_val = a[indices[:-1] + [k] + indices[-1:]]
            b_val = b[k]
        else:
            a_indices = broadcast_indices(indices[:-2], a.shape[:-2], c_shape[:-2])
            b_indices = broadcast_indices(indices[:-2], b.shape[:-2], c_shape[:-2])
            if not ta:
                a_val = a[a_indices + [indices[-2], k]]
            else:
                a_val = a[a_indices + [k, indices[-1]]]
            if not tb:
                b_val = b[b_indices + [k, indices[-1]]]
            else:
                b_val = b[b_indices + [indices[-2], k]]
        return a_val * b_val

    c = compute(
        name='c',
        shape=c_shape,
        fcompute=lambda *indices: reduce(
            shape=[reduce_extent], fcompute=lambda k: fcompute(indices, k), reduce_type='sum'
        ),
    )
    return c
