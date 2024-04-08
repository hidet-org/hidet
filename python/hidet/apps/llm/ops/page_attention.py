from typing import List, Union, Tuple

from hidet.ir import IRModule
from hidet.ir.dtypes import float16
from hidet.graph.tensor import Tensor
from hidet.graph.ops.opaque import OpaqueOperator
from hidet.ir.library import tune


class PageAttentionWriteCacheOp(OpaqueOperator):
    def __init__(self, seq_lengths, key, value, cache_slots, key_cache, value_cache):
        # seq_lengths: [bs]
        #   key: [bs, num_kv_heads, max_seq_length, head_size]
        # value: [bs, num_kv_heads, max_seq_length, head_size]
        # cache_slots: [bs, max_seq_length]
        # key_cache: [num_blocks, num_kv_heads, head_size, block_size]
        # value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        super().__init__(
            name='cache_write',
            inputs={
                'seq_lengths': seq_lengths,
                'key': key,
                'value': value,
                'cache_slots': cache_slots,
                'key_cache': key_cache,
                'value_cache': value_cache,
            },
            share_map={0: 4, 1: 5},  # share key_cache and value_cache
        )

    def symbolic_forward(self, seq_lengths, key, value, cache_slots, key_cache, value_cache):
        return {'key_cache_out': key_cache, 'value_cache_out': value_cache}

    def implement_cuda(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_cuda)

    @tune.space(1)  # empty space
    def schedule_cuda(self):
        import hidet
        from hidet.lang import attrs
        from hidet.lang.types import i32, f16, i64, shared_tensor
        from hidet.lang.cuda import blockIdx, threadIdx, syncthreads
        from hidet.lang.mapping import spatial

        bs, num_kv_heads, max_seq_length, head_size = self.inputs[1].shape
        num_blocks, num_kv_heads, head_size, block_size = self.inputs[4].shape

        with hidet.script_module() as script_module:
            seq_tile = 1
            dim_tile = 1
            assert int(head_size % (dim_tile * 4)) == 0
            assert int(block_size % (seq_tile * 4)) == 0

            @hidet.script
            def _prefill_cache_write(
                seq_lengths: i32[bs],
                inp: f16[bs, num_kv_heads, max_seq_length, head_size],
                cache_slots: i64[bs, max_seq_length],
                cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                buf: f16[seq_tile, dim_tile * 4 + 1],
            ):
                attrs.func_kind = 'cuda_internal'

                bs_idx = blockIdx.x
                kv_head_idx = blockIdx.y
                seq_length = seq_lengths[bs_idx]

                for t in range((seq_length + seq_tile - 1) // seq_tile):
                    if (t + 1) * seq_tile < seq_length:
                        # do not need to check boundary
                        for j in range(head_size // (dim_tile * 4)):
                            # read to buf [seq_tile, dim_tile * 4]
                            for i, jj in spatial(seq_tile, dim_tile).on(threadIdx.x):
                                for jjj in range(4):
                                    seq_idx = t * seq_tile + i
                                    dim_idx = j * (dim_tile * 4) + jj * 4 + jjj
                                    buf[i, jj * 4 + jjj] = inp[bs_idx, kv_head_idx, seq_idx, dim_idx]
                            syncthreads()

                            # write buf to cache in global memory
                            for jj, ii in spatial(dim_tile, seq_tile).on(threadIdx.x):
                                seq_idx = t * seq_tile + ii
                                cache_slot = cache_slots[bs_idx, seq_idx]
                                block_idx = i32(cache_slot // block_size)
                                block_offset = i32(cache_slot % block_size)
                                for jjj in range(4):
                                    dim_idx = j * dim_tile * 4 + jj * 4 + jjj
                                    cache[block_idx, kv_head_idx, dim_idx, block_offset] = buf[ii, jj * 4 + jjj]
                            syncthreads()
                    else:
                        # do not need to check boundary
                        for j in range(head_size // (dim_tile * 4)):
                            # read to buf [seq_tile, dim_tile * 4]
                            for i, jj in spatial(seq_tile, dim_tile).on(threadIdx.x):
                                for jjj in range(4):
                                    seq_idx = t * seq_tile + i
                                    dim_idx = j * (dim_tile * 4) + jj * 4 + jjj
                                    if seq_idx < seq_length:
                                        buf[i, jj * 4 + jjj] = inp[bs_idx, kv_head_idx, seq_idx, dim_idx]
                            syncthreads()

                            # write buf to cache in global memory
                            for jj, ii in spatial(dim_tile, seq_tile).on(threadIdx.x):
                                seq_idx = t * seq_tile + ii
                                if seq_idx < seq_length:
                                    cache_slot = cache_slots[bs_idx, seq_idx]
                                    block_idx = i32(cache_slot // block_size)
                                    block_offset = i32(cache_slot % block_size)
                                    for jjj in range(4):
                                        dim_idx = j * dim_tile * 4 + jj * 4 + jjj
                                        cache[block_idx, kv_head_idx, dim_idx, block_offset] = buf[ii, jj * 4 + jjj]
                            syncthreads()

            @hidet.script
            def prefill_cache_write(
                seq_lengths: i32[bs],
                key: f16[bs, num_kv_heads, max_seq_length, head_size],
                value: f16[bs, num_kv_heads, max_seq_length, head_size],
                cache_slots: i64[bs, max_seq_length],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = bs, num_kv_heads
                attrs.cuda.block_dim = seq_tile * dim_tile

                buf = shared_tensor(dtype=f16, shape=[seq_tile, dim_tile * 4 + 1])

                _prefill_cache_write(seq_lengths, key, cache_slots, key_cache, buf)
                _prefill_cache_write(seq_lengths, value, cache_slots, value_cache, buf)

            @hidet.script
            def decode_cache_write(
                seq_lengths: i32[bs],
                key: f16[bs, num_kv_heads, 1, head_size],
                value: f16[bs, num_kv_heads, 1, head_size],
                cache_slots: i64[bs, 1],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = bs, num_kv_heads
                attrs.cuda.block_dim = head_size

                bs_idx = blockIdx.x
                kv_head_idx = blockIdx.y
                dim_idx = threadIdx.x

                # get cache slot
                cache_slot = cache_slots[bs_idx, 0]
                block_idx = cache_slot / block_size
                block_offset = cache_slot % block_size

                # store key and value to cache
                key_cache[block_idx, kv_head_idx, dim_idx, block_offset] = key[bs_idx, kv_head_idx, 0, dim_idx]
                value_cache[block_idx, kv_head_idx, dim_idx, block_offset] = value[bs_idx, kv_head_idx, 0, dim_idx]

            @hidet.script
            def launch(
                seq_lengths: i32[bs],
                key: f16[bs, num_kv_heads, max_seq_length, head_size],
                value: f16[bs, num_kv_heads, max_seq_length, head_size],
                cache_slots: i64[bs, max_seq_length],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                key_cache_out: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache_out: f16[num_blocks, num_kv_heads, head_size, block_size],
            ):
                attrs.func_kind = 'public'

                if max_seq_length == 1:
                    decode_cache_write(seq_lengths, key, value, cache_slots, key_cache, value_cache)
                else:
                    prefill_cache_write(seq_lengths, key, value, cache_slots, key_cache, value_cache)

        return script_module.ir_module()


class PageAttentionOp(OpaqueOperator):
    def __init__(
        self, query: Tensor, seq_lengths: Tensor, cache_blocks: Tensor, key_cache: Tensor, value_cache: Tensor
    ):
        super().__init__(
            name='page_attention',
            inputs={
                'query': query,
                'seq_lengths': seq_lengths,
                'cache_blocks': cache_blocks,
                'key_cache': key_cache,
                'value_cache': value_cache,
            },
        )

    def symbolic_forward(
        self, query: Tensor, seq_lengths: Tensor, cache_blocks: Tensor, key_cache: Tensor, value_cache: Tensor
    ):
        assert query.dtype == key_cache.dtype == value_cache.dtype, 'Mismatched dtype of query, key, value'
        assert query.dtype in [float16], f'Unsupported dtype: {query.dtype}'
        bs, num_heads, _, head_size = query.shape
        return {'output': self.symbol(shape=[bs, num_heads, 1, head_size], dtype=query.dtype, device=query.device)}

    def implement_cuda(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_cuda)

    @tune.space(1)  # empty space
    def schedule_cuda(self) -> IRModule:
        # naive implementation, todo: optimize this kernel
        import hidet
        from hidet.lang import attrs, cast
        from hidet.lang.types import u8, f16, f32, i32, tensor_pointer_type, tensor_pointer
        from hidet.lang.cuda import memcpy, blockIdx, threadIdx, shfl_down_sync, shfl_sync, blockDim
        from hidet.ir.primitives.math import exp, sqrt
        from hidet.ir.primitives import runtime

        _query, _seq_lengths, _cache_blocks, _key_cache, _value_cache = self.inputs
        bs, num_heads, _, head_size = self.inputs[0].shape
        max_cache_blocks = _cache_blocks.shape[-1]
        num_blocks, num_kv_heads, head_size, block_size = _key_cache.shape

        tile_size = 128

        assert int(32 % block_size) == 0
        with hidet.script_module() as script_module:

            @hidet.script
            def page_attention_score(
                max_seq_length: i32,
                score_ptr: ~f32,
                query: f16[bs, num_heads, head_size],
                seq_lengths: i32[bs],
                cache_blocks: i32[bs, max_cache_blocks],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = (max_seq_length + tile_size - 1) // tile_size, num_heads, bs
                attrs.cuda.block_dim = tile_size

                bs_idx = blockIdx.z
                head_idx = blockIdx.y

                score = tensor_pointer(f32, [bs, num_heads, max_seq_length], init=score_ptr)

                j = blockIdx.x * tile_size + threadIdx.x
                seq_length = seq_lengths[bs_idx]

                if j < seq_length:
                    acc = f16.zero
                    block_idx = cache_blocks[bs_idx, j // block_size]
                    block_offset = j % block_size
                    kv_head_idx = head_idx % num_kv_heads
                    for k in range(head_size):
                        a = query[bs_idx, head_idx, k]
                        b = key_cache[block_idx, kv_head_idx, k, block_offset]
                        acc += a * b
                    acc = acc / sqrt(cast(head_size, f32))
                    score[bs_idx, head_idx, j] = acc

            @hidet.script
            def warp_max(val: f32) -> f32:
                attrs.func_kind = 'cuda_internal'
                for i in range(5):
                    val = max(val, shfl_down_sync(0xFFFFFFFF, val, 1 << i))
                val = shfl_sync(0xFFFFFFFF, val, 0)
                return val

            @hidet.script
            def warp_sum(val: f32) -> f32:
                attrs.func_kind = 'cuda_internal'
                for i in range(5):
                    val = val + shfl_down_sync(0xFFFFFFFF, val, 1 << i)
                val = shfl_sync(0xFFFFFFFF, val, 0)
                return val

            @hidet.script
            def page_attention_softmax(max_seq_length: i32, output_ptr: ~f32, score_ptr: ~f32, seq_lengths: i32[bs]):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = num_heads, bs
                attrs.cuda.block_dim = 32

                output = tensor_pointer(f32, [bs, num_heads, max_seq_length], init=output_ptr)
                score = tensor_pointer(f32, [bs, num_heads, max_seq_length], init=score_ptr)

                bs_idx = blockIdx.y
                head_idx = blockIdx.x

                seq_length = seq_lengths[bs_idx]
                warp_size = 32

                # max value
                max_val = f32.min_value
                for i in range((seq_length + warp_size - 1) / warp_size):
                    j = i * blockDim.x + threadIdx.x
                    if j < seq_length:
                        max_val = max(max_val, score[bs_idx, head_idx, j])
                max_val = warp_max(max_val)

                # sum exp
                sum_exp = f32.zero
                for i in range((seq_length + warp_size - 1) / warp_size):
                    j = i * blockDim.x + threadIdx.x
                    if j < seq_length:
                        sum_exp += exp(score[bs_idx, head_idx, j] - max_val)
                sum_exp = warp_sum(sum_exp)

                # divide
                for i in range((seq_length + warp_size - 1) / warp_size):
                    j = i * blockDim.x + threadIdx.x
                    if j < seq_length:
                        output[bs_idx, head_idx, j] = exp(score[bs_idx, head_idx, j] - max_val) / sum_exp

            @hidet.script
            def page_attention_output(
                max_seq_length: i32,
                output: f16[bs, num_heads, 1, head_size],
                score_ptr: ~f32,
                seq_lengths: i32[bs],
                cache_blocks: i32[bs, max_cache_blocks],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = num_heads, bs
                attrs.cuda.block_dim = head_size

                bs_idx = blockIdx.y
                head_idx = blockIdx.x
                kv_head_idx = head_idx % num_kv_heads

                score = tensor_pointer(f32, [bs, num_heads, max_seq_length], init=score_ptr)

                j = threadIdx.x
                seq_length = seq_lengths[bs_idx]

                acc = f32.zero

                for k in range(seq_length):
                    a = score[bs_idx, head_idx, k]
                    block_idx = cache_blocks[bs_idx, k // block_size]
                    block_offset = k % block_size
                    b = value_cache[block_idx, kv_head_idx, j, block_offset]
                    acc += a * b

                output[bs_idx, head_idx, 0, j] = cast(acc, f16)

            @hidet.script
            def launch(
                query: f16[bs, num_heads, 1, head_size],
                seq_lengths: i32[bs],
                cache_blocks: i32[bs, max_cache_blocks],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                output: f16[bs, num_heads, 1, head_size],
            ):
                attrs.func_kind = 'public'

                # calculate max_seq_length
                max_seq_length: i32 = 0
                seq_lengths_cpu = cast(
                    runtime.request_cpu_workspace(nbytes=bs * i32.nbytes), dtype=tensor_pointer_type(i32, [bs])
                )
                memcpy(dst=seq_lengths_cpu, src=seq_lengths, count=bs * i32.nbytes, kind='cuda_to_cpu')
                for i in range(bs):
                    max_seq_length = max(max_seq_length, seq_lengths_cpu[i])

                # allocate cuda buffers
                cuda_workspace = cast(
                    runtime.request_cuda_workspace(nbytes=2 * bs * num_heads * max_seq_length * f32.nbytes), dtype=~u8
                )
                score = cast(~cuda_workspace[0], dtype=tensor_pointer_type(f32, [bs, num_heads, max_seq_length]))
                softmax = cast(
                    ~cuda_workspace[bs * num_heads * max_seq_length * f32.nbytes],
                    dtype=tensor_pointer_type(f32, [bs, num_heads, max_seq_length]),
                )

                # score = query @ key / sqrt(head_size)
                page_attention_score(max_seq_length, score, query, seq_lengths, cache_blocks, key_cache)

                # softmax(score)
                page_attention_softmax(max_seq_length, softmax, score, seq_lengths)

                # output = softmax @ value
                page_attention_output(max_seq_length, output, softmax, seq_lengths, cache_blocks, value_cache)

        return script_module.ir_module()


def cache_write(
    seq_lengths: Tensor, key: Tensor, value: Tensor, cache_slots: Tensor, key_cache: Tensor, value_cache: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Write the key and value to the cache.

    Parameters
    ----------
    seq_lengths: Tensor
        The sequence lengths. Shape: i32 [bs]
    key: Tensor
        The key tensor. Shape: [bs, num_kv_heads, max_seq_length, head_size]
    value: Tensor
        The value tensor. Shape: [bs, num_kv_heads, max_seq_length, head_size]
    cache_slots: Tensor
        The cache slots. Shape: i64 [bs, max_seq_length]
    key_cache: Tensor
        The key cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]
    value_cache: Tensor
        The value cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]

    Returns
    -------
    (updated_key_cache, updated_value_cache): Tuple[Tensor, Tensor]
        updated_key_cache: The updated key cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]
        updated_value_cache: The updated value cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]
    """
    return PageAttentionWriteCacheOp(seq_lengths, key, value, cache_slots, key_cache, value_cache).outputs


def page_attention(query: Tensor, seq_lengths: Tensor, cache_blocks: Tensor, key_cache: Tensor, value_cache: Tensor):
    """
    Page attention.

    Parameters
    ----------
    query: Tensor
        The query tensor. Shape: [bs, num_heads, 1, head_size]

    seq_lengths: Tensor
        The sequence lengths. Shape: [bs]

    cache_blocks: Tensor
        The cache slots. Shape: [bs, max_cache_blocks]

    key_cache: Tensor
        The key cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]

    value_cache: Tensor
        The value cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]

    Returns
    -------
    output: Tensor
        The output tensor. Shape: [bs, num_heads, 1, head_size]
    """
    return PageAttentionOp(query, seq_lengths, cache_blocks, key_cache, value_cache).outputs[0]
