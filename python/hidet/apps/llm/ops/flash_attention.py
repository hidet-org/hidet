from hidet.graph.tensor import Tensor
from hidet.graph import ops


def flash_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    Flash attention.

    Parameters
    ----------
    query: Tensor
        The query tensor. Shape: [bs, num_heads, seq_length, head_size]

    key: Tensor
        The key tensor. Shape: [bs, num_kv_heads, seq_length, head_size]

    value: Tensor
        The value tensor. Shape: [bs, num_kv_heads, seq_length, head_size]

    Returns
    -------
    output: Tensor
        The output tensor. Shape: [bs, num_heads, seq_length, head_size]
    """
    from hidet.ir.expr import cast

    seq_length = query.shape[-2]
    transposed_key = ops.transpose(key, [0, 1, 3, 2])  # [bs, num_kv_heads, head_size, seq_length]
    norm_scalar = ops.sqrt(ops.full([], value=cast(seq_length, dtype='float16'), device=query.device))
    causal_mask = (
        1.0 - ops.tri(seq_length, seq_length, dtype=query.dtype, device=query.device)
    ) * query.dtype.min_value
    score = ops.matmul(query, transposed_key) / norm_scalar + causal_mask
    softmax_score = ops.softmax(score, axis=-1)  # [bs, num_heads, seq_length, seq_length]
    output = ops.matmul(softmax_score, value)  # [bs, num_heads, seq_length, head_size]
    return output
