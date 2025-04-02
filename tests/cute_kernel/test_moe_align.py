# SPDX-License-Identifier: Apache-2.0
# type: ignore
# ruff: noqa
# type: ignore
"""
Helper functions for Mixture of Experts (MoE) layer implementation.
This module provides utilities for handling MoE operations, particularly focusing on
block size alignment and weight quantization/dequantization for efficient MoE computation.
"""

from typing import Union

import hidet
import torch
from hidet.ir.cute import TensorLayout, TiledTensorLayout
from hidet.ir.cute.algorithm import auto_copy
from hidet.ir.cute.layout import Level, ThrValAtom, auto_layout, composition, layout_auto, logical_divide, make_layout
from hidet.ir.cute.ops import (
    cast,
    copy,
    fill,
    make_tensor,
    mask,
    partition_dst,
    partition_src,
    rearrange,
    reduce_sum,
    silu,
    tensor_view,
)
from hidet.ir.dtypes import f16, f32, i32, i64, u4, u32
from hidet.ir.expr import symbol_var
from hidet.ir.primitives.cuda.atomic import atomic_add
from hidet.ir.primitives.cuda.mutex import acquire_seq_semaphore
from hidet.ir.type import DataType, data_type
from hidet.lang import attrs
from hidet.lang.cuda import blockIdx, dynamic_shared_memory, syncthreads, threadIdx
from hidet.utils.py import cdiv


def moe_align_block_size_stage1(num_experts: int, threads: int = 256):
    """
    First stage of MoE block size alignment kernel.

    This kernel performs the initial counting and alignment of tokens per expert.
    It counts how many tokens are assigned to each expert and prepares the data
    for the second stage of alignment.

    Args:
        num_experts (int): Number of experts in the MoE layer
        threads (int, optional): Number of threads per block. Defaults to 256.

    Returns:
        Compiled CUDA kernel function
    """
    scalar_t = u32
    num_tokens = symbol_var("total_tokens")
    block_size = symbol_var("block_size")
    num_blocks = cdiv(num_tokens, threads)
    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)

    dynamic_smem_bytes = (num_experts * threads + num_experts + 1) * i32.nbytes

    accesses_per_threads = 4
    atom_shape = (1, accesses_per_threads)
    atom = TensorLayout(((1,), (1, accesses_per_threads)), ((1,), (1, 1)))
    tv_atom = ThrValAtom("thread", atom_shape, atom)
    from hidet.utils.py import gcd

    thread_n = gcd(threads // accesses_per_threads, threads)
    thread_m = threads // thread_n
    repeat_n = 1
    repeat_m = num_experts // thread_m
    threads_in_thread_block = Level(
        "thread",
        "thread_block",
        (thread_m, thread_n),
        TensorLayout((thread_n, thread_m), (thread_m, 1)),
        (repeat_m, repeat_n),
    )
    layout_tokens_cnt = TiledTensorLayout(tv_atom, [threads_in_thread_block])

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            topk_ids: scalar_t[num_tokens],
            token_cnts: i32[num_blocks, num_experts],
            lock: ~i32,
            expert_ids: i32[max_num_m_blocks],
            total_tokens_per_expert: i32[num_experts],
            expert_start_index: i32[num_experts],
            num_tokens_post_pad: ~i32,
        ):
            # Step 1: Initialize shared memory for token counts
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = num_blocks
            attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

            tid = threadIdx.x
            bid = blockIdx.x
            gid = threadIdx.x + blockIdx.x * threads

            # Allocate shared memory for token counts and cumulative sums
            smem_token_cnts = dynamic_shared_memory(byte_offset=0, dtype=i32)
            smem_cumsum = dynamic_shared_memory(byte_offset=num_experts * threads * 4, dtype=i32)

            # Step 2: Initialize token counts to zero
            for i in range(num_experts):
                smem_token_cnts[i * threads + tid] = 0
            syncthreads()

            # Step 3: Count tokens per expert
            if gid < num_tokens:
                expert_id = topk_ids[gid]
                smem_token_cnts[expert_id * threads + tid] += 1

            syncthreads()

            # Step 4: Reduce token counts across threads
            # Note: here, we need to declare the shared tensor with
            # volatile=True. We use a mixed programming paradigm
            # in this kernel. (i.e., combining the SIMD and tile-based
            # programming model) This shared memory tensor is written
            # by tile-level primitives, but the data is not read by
            # any tile-level operations and not written into global
            # memory. So it will be eliminated during deadcode
            # elimination pass if we don't declare it with `volatile=True`.
            ts_token_cnt = tensor_view(
                smem_token_cnts, TensorLayout((num_experts, threads), (threads, 1)), "shared", volatile=True
            )
            tr_token_cnt = make_tensor(i32, layout_tokens_cnt, "register")
            txstoken_cnt = partition_src(ts_token_cnt, auto_copy())
            txrtoken_cnt = partition_dst(tr_token_cnt, auto_copy())
            copy(auto_copy((num_experts, threads)), txstoken_cnt, txrtoken_cnt)

            syncthreads()

            # Step 5: Sum up token counts
            tr_token_cnt_sum = reduce_sum(tr_token_cnt, 1)

            # Step 6: Handle block synchronization and accumulation
            if bid > 0:
                # For non-first blocks, accumulate counts to global memory
                tg_token_cnt = tensor_view(token_cnts[bid, :], TensorLayout((num_experts, threads), (1, 0)), "global")
                txgtoken_cnt = partition_dst(tg_token_cnt, auto_copy())
                txrtoken_cnt_sum = partition_src(tr_token_cnt_sum, auto_copy())
                copy(auto_copy((num_experts, threads)), txrtoken_cnt_sum, txgtoken_cnt)
                syncthreads()

                if tid == 0:
                    atomic_add(lock, 1, sem="acq_rel")
            else:
                # For first block, handle synchronization and compute final indices
                acquire_seq_semaphore(lock, num_blocks - 1)

                # Accumulate counts from other blocks
                for i in range(num_blocks - 1):
                    tr_token_cnt_partial = make_tensor(i32, layout_auto((num_experts, threads), (1, 0)), "register")
                    tg_token_cnt_partial = tensor_view(
                        token_cnts[i + 1, :], TensorLayout((num_experts, threads), (1, 0)), "global"
                    )
                    txgtoken_cnt_partial = partition_src(tg_token_cnt_partial, auto_copy())
                    txrtoken_cnt_partial = partition_dst(tr_token_cnt_partial, auto_copy())
                    copy(auto_copy((num_experts, threads)), txgtoken_cnt_partial, txrtoken_cnt_partial)
                    tr_token_cnt_sum = tr_token_cnt_partial + tr_token_cnt_sum

                # Compute cumulative sums and expert indices
                ts_cumsum = tensor_view(
                    smem_cumsum + 1, TensorLayout((num_experts, threads), (1, 0)), "shared", volatile=True
                )
                txscumsum = partition_src(ts_cumsum, auto_copy())
                txrcumsum = partition_dst(tr_token_cnt_sum, auto_copy())
                copy(auto_copy((num_experts, threads)), txrcumsum, txscumsum)

                syncthreads()

                # Step 7: Compute final indices and padding
                if tid == 0:
                    smem_cumsum[0] = 0
                    for i in range(num_experts):
                        total_tokens_per_expert[i] = smem_cumsum[i + 1]
                        expert_start_index[i] = smem_cumsum[i] // block_size
                        smem_cumsum[i + 1] = (
                            smem_cumsum[i] + (smem_cumsum[i + 1] + block_size - 1) // block_size * block_size
                        )
                    num_tokens_post_pad[0] = smem_cumsum[num_experts]

                syncthreads()

                # Step 8: Generate expert IDs for each block
                rounds = cdiv(num_experts, threads)
                for i in range(rounds):
                    eid = i * threads + tid
                    if eid < num_experts:
                        beg = smem_cumsum[eid] // block_size
                        end = smem_cumsum[eid + 1] // block_size
                        for j in range(end - beg):
                            expert_ids[j + beg] = eid

    func = script_module.build()
    return func


def moe_align_block_size_stage2(num_experts: int, threads: int = 256):
    """
    Second stage of MoE block size alignment kernel.

    This kernel performs the final sorting and alignment of tokens based on expert assignments.
    It uses the results from stage 1 to place tokens in their correct positions in the sorted buffer.

    Args:
        num_experts (int): Number of experts in the MoE layer
        threads (int, optional): Number of threads per block. Defaults to 256.

    Returns:
        Compiled CUDA kernel function
    """
    scalar_t = u32
    num_tokens = symbol_var("total_tokens")
    block_size = symbol_var("block_size")
    num_blocks = cdiv(num_tokens, threads)
    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            topk_ids: scalar_t[num_tokens],
            token_cnts: i32[num_blocks, num_experts],
            expert_start_index: i32[num_experts],
            sorted_topk_ids: i32[max_num_tokens_padded],
        ):
            # Step 1: Set up kernel parameters and thread indices
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = num_blocks
            attrs.cuda.dynamic_smem_bytes = 0

            bid = blockIdx.x
            gid = threadIdx.x + blockIdx.x * threads

            # Step 2: Sort tokens by expert and compute their positions
            if gid < num_tokens:
                expert_id = topk_ids[gid]
                pos = expert_start_index[expert_id] * block_size
                ticket = atomic_add(~token_cnts[0, expert_id], 1)
                sorted_topk_ids[pos + ticket] = gid

    func = script_module.build()
    return func


class MoEAlignBlockSize:
    """
    A class that implements the two-stage MoE block size alignment process.

    This class combines the two stages of MoE block size alignment into a single interface.
    It manages the execution of both stages and handles the necessary data transfers between them.

    Args:
        num_experts (int): Number of experts in the MoE layer
        threads (int, optional): Number of threads per block. Defaults to 256.
    """

    def __init__(self, num_experts: int, threads: int = 256):
        self.func1 = moe_align_block_size_stage1(num_experts, threads)
        self.func2 = moe_align_block_size_stage2(num_experts, threads)

    def __call__(
        self,
        topk_ids: torch.Tensor,
        token_cnts: torch.Tensor,
        lock: torch.Tensor,
        expert_ids: torch.Tensor,
        sorted_topk_ids: torch.Tensor,
        total_tokens_per_expert: torch.Tensor,
        expert_start_index: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
    ):
        """
        Execute the two-stage MoE block size alignment process.

        Args:
            topk_ids (torch.Tensor): Input tensor containing expert assignments for each token
            token_cnts (torch.Tensor): Counter tensor for tracking tokens per expert
            lock (torch.Tensor): Synchronization lock for inter-block coordination
            expert_ids (torch.Tensor): Output tensor for expert IDs
            sorted_topk_ids (torch.Tensor): Output tensor for sorted token IDs
            total_tokens_per_expert (torch.Tensor): Output tensor for total tokens per expert
            expert_start_index (torch.Tensor): Output tensor for expert start indices
            num_tokens_post_pad (torch.Tensor): Output tensor for number of padded tokens
        """
        self.func1(
            topk_ids, token_cnts, lock, expert_ids, total_tokens_per_expert, expert_start_index, num_tokens_post_pad
        )
        self.func2(topk_ids, token_cnts, expert_start_index, sorted_topk_ids)


def moe_align_block_size_kernel(num_experts: int, threads: int = 256):
    """
    Factory function to create a MoEAlignBlockSize instance.

    Args:
        num_experts (int): Number of experts in the MoE layer
        threads (int, optional): Number of threads per block. Defaults to 256.

    Returns:
        MoEAlignBlockSize: An instance of the MoEAlignBlockSize class
    """
    return MoEAlignBlockSize(num_experts, threads)


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """
    Main function to perform MoE block size alignment.

    This function orchestrates the two-stage block size alignment process for MoE computation.
    It prepares the necessary tensors and executes the alignment kernels.

    Args:
        topk_ids (torch.Tensor): Input tensor containing expert assignments for each token
        block_size (int): Size of blocks for alignment
        num_experts (int): Number of experts in the MoE layer

    Returns:
        tuple: A tuple containing:
            - sorted_ids (torch.Tensor): Sorted token IDs aligned by expert
            - expert_ids (torch.Tensor): Expert IDs for each block
            - total_tokens_per_expert (torch.Tensor): Total tokens assigned to each expert
            - expert_start_index (torch.Tensor): Starting index for each expert's tokens
            - num_tokens_post_pad (torch.Tensor): Number of padded tokens after alignment
    """
    NUM_THREADS_PER_BLOCK = 64
    threads = NUM_THREADS_PER_BLOCK
    num_tokens = topk_ids.numel()
    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)
    from hidet.ffi import runtime_api

    runtime_api.set_symbol_value("total_tokens", num_tokens)
    runtime_api.set_symbol_value("block_size", block_size)
    func = moe_align_block_size_kernel(num_experts, threads)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    total_tokens_per_expert = torch.empty((num_experts,), dtype=torch.int32, device=topk_ids.device)
    expert_start_index = torch.empty((num_experts,), dtype=torch.int32, device=topk_ids.device)
    lock = torch.zeros((1,), dtype=torch.int32, device=topk_ids.device)
    tokens_cnt = torch.zeros((cdiv(num_tokens, threads), num_experts), dtype=torch.int32, device=topk_ids.device)
    func(
        topk_ids,
        tokens_cnt,
        lock,
        expert_ids,
        sorted_ids,
        total_tokens_per_expert,
        expert_start_index,
        num_tokens_post_pad,
    )
    return sorted_ids, expert_ids, total_tokens_per_expert, expert_start_index, num_tokens_post_pad


def test_moe_align_block_size():
    with hidet.option.context():
        # hidet.option.cache_dir("moe")
        # hidet.option.debug_cache_tuning(True)
        # hidet.option.save_lower_ir(True)
        num_experts = 256
        topk = 8
        num_tokens = 32
        block_size = 16
        max_num_tokens_padded = num_tokens * topk + num_experts * (block_size - 1)
        max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)
        tokens = num_tokens * topk
        topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.uint32, device="cuda")
        sorted_ids, expert_ids, total_tokens_per_expert, expert_start_index, num_tokens_post_pad = moe_align_block_size(
            topk_ids, block_size, num_experts
        )
        print(topk_ids)
        print(sorted_ids)
        for i, id in enumerate(sorted_ids):
            bid = i // block_size
            expert_id = expert_ids[bid]
            token_id = id // topk
            expert_id_ = topk_ids[token_id, id % topk]
            assert id.item() == 0 or expert_id.item() == expert_id_.item()
        exps = {}
        for i in range(num_tokens):
            for j in range(topk):
                expert_id = topk_ids[i, j].item()
                if expert_id not in exps:
                    exps[expert_id] = 0
                exps[expert_id] += 1
        print(expert_ids)
        print(total_tokens_per_expert)
        for i, id in enumerate(total_tokens_per_expert):
            expert_id = i
            num_tokens = id.item()
            if num_tokens == 0:
                continue
            num_tokens_ = exps[expert_id]
            assert num_tokens_ == num_tokens
        print(expert_start_index)
        print(num_tokens_post_pad)
