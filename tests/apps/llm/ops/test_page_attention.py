from typing import List
import pytest
import math
import torch

import hidet.utils
import hidet.option
from hidet import Tensor, from_torch
from hidet.apps.llm.ops.page_attention import page_attention


def page_attention_ref(
    query: Tensor, seq_lengths: Tensor, cache_blocks: Tensor, key_cache: Tensor, value_cache: Tensor
):
    query = query.torch()
    seq_lengths = seq_lengths.torch()
    cache_blocks = cache_blocks.torch()
    key_cache = key_cache.torch()
    value_cache = value_cache.torch()

    bs = query.size(0)
    num_heads = query.size(1)
    num_kv_heads = key_cache.size(1)
    head_size = query.size(3)
    block_size = key_cache.size(-1)
    outputs = []

    for i in range(bs):
        seq_query = query[i, :, 0, :]  # [num_heads, head_size]
        seq_key_list = []
        seq_value_list = []
        seq_length = int(seq_lengths[i])
        num_blocks = (seq_length + block_size - 1) // block_size
        for j in range(num_blocks):
            block_idx = int(cache_blocks[i, j])
            seq_key_list.append(key_cache[block_idx, :, :, :])  # [num_kv_heads, head_size, block_size]
            seq_value_list.append(value_cache[block_idx, :, :, :])  # [num_kv_heads, head_size, block_size]
        seq_key = torch.cat(seq_key_list, dim=-1)[:, :, :seq_length]  # [num_kv_heads, head_size, seq_length]
        seq_value = torch.cat(seq_value_list, dim=-1)[:, :, :seq_length]  # [num_kv_heads, head_size, seq_length]
        if num_heads != num_kv_heads:
            assert num_heads % num_kv_heads == 0
            seq_key = seq_key.repeat(num_heads // num_kv_heads, 1, 1)  # [num_heads, head_size, seq_length]
            seq_value = seq_value.repeat(num_heads // num_kv_heads, 1, 1)  # [num_heads, head_size, seq_length]

        seq_query = seq_query.unsqueeze(1)  # [num_heads, 1, head_size]
        seq_value = seq_value.transpose(1, 2)  # [num_heads, seq_length, head_size]
        score = torch.matmul(seq_query, seq_key) / math.sqrt(head_size)  # [num_heads, 1, seq_length]
        score = torch.softmax(score, dim=-1)  # [num_heads, 1, seq_length]
        output = torch.matmul(score, seq_value)  # [num_heads, 1, head_size]
        output = torch.unsqueeze(output, dim=0)  # [1, num_heads, 1, head_size]
        outputs.append(output)
    output = torch.cat(outputs, dim=0)  # [bs, num_heads, 1, head_size]
    return from_torch(output)


@pytest.mark.parametrize('num_heads, num_kv_heads', [(32, 1), (32, 32), (64, 8)])
@pytest.mark.parametrize('block_size, head_size', [(4, 32), (16, 64), (32, 128)])
@pytest.mark.parametrize('seq_lengths_list', [[1, 2, 300, 448, 5, 683, 791, 88, 9]])
def test_page_attention(num_heads, num_kv_heads, block_size, head_size, seq_lengths_list: List[int]):
    bs = len(seq_lengths_list)
    cache_blocks_list = []
    max_num_blocks = max((seq_length + block_size - 1) // block_size for seq_length in seq_lengths_list)

    # generate cache_blocks
    current_block = 0
    for seq_length in seq_lengths_list:
        num_blocks = (seq_length + block_size - 1) // block_size
        blocks = []
        for _ in range(num_blocks):
            blocks.append(current_block)
            current_block += 1
        while len(blocks) < max_num_blocks:
            blocks.append(-1)
        cache_blocks_list.append(blocks)
    num_blocks = current_block

    # generate inputs
    query = from_torch(torch.randn(bs, num_heads, 1, head_size, dtype=torch.float16, device='cuda'))
    seq_lengths = from_torch(torch.tensor(seq_lengths_list, dtype=torch.int32, device='cuda'))
    cache_blocks = from_torch(torch.tensor(cache_blocks_list, dtype=torch.int32, device='cuda'))
    key_cache = from_torch(
        torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.float16, device='cuda')
    )
    value_cache = from_torch(
        torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.float16, device='cuda')
    )

    # run reference implementation
    output_ref = page_attention_ref(query, seq_lengths, cache_blocks, key_cache, value_cache)
    torch.cuda.synchronize()

    # run our implementation
    output_our = page_attention(query, seq_lengths, cache_blocks, key_cache, value_cache)
    torch.cuda.synchronize()

    hidet.utils.assert_close(output_our, output_ref, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
