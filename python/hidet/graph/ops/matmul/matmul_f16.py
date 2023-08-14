# %%
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Tuple
from hidet.ir import dtypes
from hidet.ir.dtypes import float16
from hidet.ir.expr import if_then_else, Int, Expr
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.compute import TensorNode
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.utils import input_like, broadcast_shape, can_mutually_broadcast
from hidet.ir.library import tune
from hidet.graph.operator import Operator, Tensor
from hidet.utils.py import is_power_of_two, cdiv, prod
from hidet.graph.ops.utils import broadcast_indices


class MatmulF16Task(Task):
    def __init__(self, a: TensorNode, b: TensorNode, parallel_k_parts: int = 1):
        if not a.type.dtype == float16 or not b.type.dtype == float16:
            raise ValueError('Both inputs must be float16 tensors')

        if len(a.shape) < 2 or len(b.shape) < 2:
            raise ValueError('Matrix multiplication expect at least 2D tensor, got {} and {}'.format(a.shape, b.shape))

        self._assert(
            a.shape[-1] == b.shape[-2],
            msg=(
                'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                ', got {} and {}'.format(a.shape, b.shape)
            ),
        )

        self._assert(
            can_mutually_broadcast(a.shape[:-2], b.shape[:-2]),
            msg=(
                'Matrix multiplication expect tensor A and B with compatible broadcast shape, '
                'got {} and {}'.format(a.shape, b.shape)
            ),
        )

        a_shape = a.shape
        b_shape = b.shape
        k_size = a.shape[-1]
        c_shape = [parallel_k_parts] + broadcast_shape(a.shape[:-2], b.shape[:-2]) + [a_shape[-2], b_shape[-1]]
        k_part_extent = cdiv(k_size, parallel_k_parts)

        c = compute(
            name='c',
            shape=c_shape,
            fcompute=lambda k_part, *indices: reduce(
                shape=[k_part_extent],
                fcompute=lambda k: if_then_else(
                    k_part * k_part_extent + k < k_size,
                    a[broadcast_indices(indices[:-2], a.shape[:-2], c_shape[1:-2]) + [indices[-2], k]]
                    * b[broadcast_indices(indices[:-2], b.shape[:-2], c_shape[1:-2]) + [k, indices[-1]]],
                    float16(0.0),
                ),
                reduce_type='sum',
            ),
        )

        super().__init__(
            name='matmul_f16_pk', inputs=[a, b], outputs=[c], attributes={'parallel_k_parts': parallel_k_parts}
        )

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> List[IRModule]:
        return tune.extract_ir_modules(self.schedule)

    @tune.space(
        2,
        block_m=[32, 64, 128, 256],
        block_n=[32, 64, 128, 256],
        block_k=[8, 16, 32, 64, 128],
        warp_m=[16, 32, 48, 64],
        warp_n=[16, 32, 48, 64],
        warp_k=[8, 16, 32, 64],
        mma=['m16n8k16'],
    )
    @tune.space(1, block_m=[128], block_n=[128], block_k=[16], warp_m=[64], warp_n=[64], warp_k=[16], mma=['m16n8k16'])
    def schedule(
        self, block_m=64, block_n=128, block_k=16, warp_m=32, warp_n=64, warp_k=16, mma: str = 'm16n8k16'
    ) -> IRModule:
        # pylint: disable=unused-variable
        import hidet
        from hidet.ir.type import tensor_type
        from hidet.lang import attrs, col_spatial, view, u32, tensor_pointer, grid
        from hidet.lang.layout import row_major
        from hidet.lang.mapping import spatial, auto_map
        from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory
        from hidet.lang.cuda import MmaConfig, mma_sync, cp_async, cp_async_wait_all, ldmatrix
        from hidet.lang.cuda import register_tensor

        # input shapes
        node_a, node_b, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape: Tuple[Int, ...] = node_a.shape
        b_shape: Tuple[Int, ...] = node_b.shape
        c_shape: Tuple[Int, ...] = node_c.shape
        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]
        a_head, b_head, c_head = list(a_shape[:-2]), list(b_shape[:-2]), list(c_shape[:-2])
        k_parts = self.attrs['parallel_k_parts']
        k_part_extent = cdiv(cdiv(k_size, k_parts), 8) * 8

        # schedule parameters
        mma_configs = {'m16n8k8': MmaConfig.m16n8k8_f16_f16(), 'm16n8k16': MmaConfig.m16n8k16_f16_f16()}
        tune.check(mma in mma_configs)
        mma_config = mma_configs[mma]

        mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
        warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
        mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
        threads = warp_count_m * warp_count_n * warp_count_k * 32
        grid_dim: Tuple[Int, Int, Int] = cdiv(m_size, block_m), cdiv(n_size, block_n), prod(c_head)
        dynamic_smem_bytes = max(2 * (block_m + block_n) * block_k * 2, block_m * block_n * 2)

        tune.check(block_m % warp_m == block_n % warp_n == block_k % warp_k == 0, 'warp dims divide block dims')
        tune.check(warp_m % mma_m == warp_n % mma_n == warp_k % mma_k == 0, 'mma dims divide warp dims')
        tune.check(threads <= 1024, 'threads in a block <= 1024')
        maximum_smem_bytes = 49152
        tune.check(dynamic_smem_bytes <= maximum_smem_bytes, 'dynamic shared memory <= 49152')

        tune.check(block_n % 64 == 0, 'block_n must be multiple of 64, required by async gmem -> smem loading')
        tune.check(block_k % 8 == 0)
        tune.check(is_power_of_two(block_k // 8))
        smem_a_type = tensor_type(
            'float16', shape=[block_m, block_k], layout=row_major(block_m, block_k // 8).swizzle(1) * row_major(1, 8)
        )
        smem_b_type = tensor_type(
            'float16',
            shape=[block_k, block_n],
            layout=row_major(block_k // 8, block_n // 64) * row_major(8, 8).swizzle(1) * row_major(1, 8),
        )
        load_smem_a_map = auto_map(block_m, block_k // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg))
        load_smem_b_map = auto_map(block_k, block_n // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg))
        store_smem_c_map = auto_map(block_m, block_n, workers=threads, on_fail=lambda msg: tune.check(False, msg))

        with hidet.script_module() as module:

            @hidet.script
            def load_regs_a(mi: int, k1: int, smem_a: smem_a_type, regs_a: float16[mma_config.a_elements]):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    row_addr = ~smem_a[wi * warp_m + mi * mma_m + p, wk * warp_k + k1 * mma_k + q * 8]
                    b32_regs = view(regs_a, u32[4])
                    ldmatrix(
                        regs=[b32_regs[0], b32_regs[1], b32_regs[2], b32_regs[3]],
                        smem_addr=row_addr,
                        shared_space_addr=False,
                        trans=False,
                    )

            @hidet.script
            def load_regs_b(mj: int, k1: int, smem_b: smem_b_type, regs_b: float16[mma_config.b_elements]):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    # have not used q as we only use the address of the first 16 threads to load 2 of 8x8 f16 matrix.
                    row_addr = ~smem_b[wk * warp_k + k1 * mma_k + p, wj * warp_n + mj * mma_n]
                    regs = view(regs_b, u32[2])
                    ldmatrix(regs=[regs[0], regs[1]], smem_addr=row_addr, trans=True)

            @hidet.script
            def warp_mma(
                regs_a: float16[mma_config.a_elements],
                regs_b: float16[mma_config.b_elements],
                regs_c: float16[mma_config.c_elements],
            ):
                mma_sync(mma_config, regs_a, regs_b, regs_c)

            @hidet.script
            def load_smem_a(k0: int, a: float16[a_head + [m_size, k_size]], smem_a: smem_a_type):
                c_head_index = spatial(*c_head).map(blockIdx.z)
                offset_m = blockIdx.x * block_m
                offset_k = c_head_index[0] * k_part_extent + k0 * block_k
                maximum_k = min(k_size, (c_head_index[0] + 1) * k_part_extent)
                gmem_a = a[broadcast_indices(c_head_index[1:], a_head, c_head[1:])][offset_m:, offset_k:]
                for i, k_seg in load_smem_a_map.on(threadIdx.x):
                    k = k_seg * 8
                    src_size = (
                        0
                        if (offset_m + i >= m_size or offset_k + k >= maximum_k)
                        else min(maximum_k - (offset_k + k), 8)
                    )
                    if a_shape[-1] % 8 == 0:
                        cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=16, src_size=src_size * 2, cache_level='global')
                    # trivially support other cp_sizes, perhaps do this in a more clever way?
                    elif a_shape[-1] % 4 == 0:
                        cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=8, src_size=min(8, src_size * 2))
                        cp_async(~smem_a[i, k + 4], ~gmem_a[i, k + 4], cp_size=8, src_size=max(0, src_size * 2 - 8))
                    elif a_shape[-1] % 2 == 0:
                        cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=4, src_size=min(4, src_size * 2))
                        cp_async(
                            ~smem_a[i, k + 2], ~gmem_a[i, k + 2], cp_size=4, src_size=min(4, max(0, src_size * 2 - 4))
                        )
                        cp_async(
                            ~smem_a[i, k + 4], ~gmem_a[i, k + 4], cp_size=4, src_size=min(4, max(0, src_size * 2 - 8))
                        )
                        cp_async(
                            ~smem_a[i, k + 6], ~gmem_a[i, k + 6], cp_size=4, src_size=min(4, max(0, src_size * 2 - 12))
                        )

            @hidet.script
            def load_smem_b(k0: int, b: float16[b_head + [k_size, n_size]], smem_b: smem_b_type):
                c_head_index = spatial(*c_head).map(blockIdx.z)
                offset_n = blockIdx.y * block_n
                offset_k = c_head_index[0] * k_part_extent + k0 * block_k
                maximum_k = min(k_size, (c_head_index[0] + 1) * k_part_extent)
                gmem_b = b[broadcast_indices(c_head_index[1:], b_head, c_head[1:])][offset_k:, offset_n:]
                for k, j_seg in load_smem_b_map.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = (
                        0 if (offset_k + k >= maximum_k or offset_n + j >= n_size) else min(n_size - (offset_n + j), 8)
                    )
                    if b_shape[-1] % 8 == 0:
                        cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=16, src_size=src_size * 2, cache_level='global')
                    # trivially support other cp_sizes, perhaps do this in a more clever way?
                    elif b_shape[-1] % 4 == 0:
                        cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=8, src_size=min(8, src_size * 2))
                        cp_async(~smem_b[k, j + 4], ~gmem_b[k, j + 4], cp_size=8, src_size=max(0, src_size * 2 - 8))
                    elif b_shape[-1] % 2 == 0:
                        cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=4, src_size=min(4, src_size * 2))
                        cp_async(
                            ~smem_b[k, j + 2], ~gmem_b[k, j + 2], cp_size=4, src_size=min(4, max(0, src_size * 2 - 4))
                        )
                        cp_async(
                            ~smem_b[k, j + 4], ~gmem_b[k, j + 4], cp_size=4, src_size=min(4, max(0, src_size * 2 - 8))
                        )
                        cp_async(
                            ~smem_b[k, j + 6], ~gmem_b[k, j + 6], cp_size=4, src_size=min(4, max(0, src_size * 2 - 12))
                        )

            @hidet.script
            def matmul_f16_kernel(
                a: float16[a_head + [m_size, k_size]],
                b: float16[b_head + [k_size, n_size]],
                c: float16[c_head + [m_size, n_size]],
            ):
                # matrix multiplication, using mma instruction
                attrs.cuda.grid_dim = grid_dim
                attrs.cuda.block_dim = threads
                # the second 2 means '2 bytes per float16'
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes
                # smem_storage = dyn_smem_storage
                smem_a = tensor_pointer(
                    'float16', shape=[2, block_m, block_k], layout=row_major(2) + smem_a_type.layout
                )
                smem_b = tensor_pointer(
                    'float16', shape=[2, block_k, block_n], layout=row_major(2) + smem_b_type.layout
                )
                smem_a = dynamic_shared_memory(byte_offset=0, dtype=float16)
                smem_b = dynamic_shared_memory(byte_offset=2 * block_m * block_k * 2, dtype=float16)
                regs_a = register_tensor(float16, [2, mma_count_m, mma_config.a_elements])
                regs_b = register_tensor(float16, [2, mma_count_n, mma_config.b_elements])
                regs_c = register_tensor(float16, [mma_count_m, mma_count_n, mma_config.c_elements])

                for i, j, p in grid(mma_count_m, mma_count_n, mma_config.c_elements):
                    regs_c[i, j, p] = 0.0

                load_smem_a(0, a, ~smem_a[0, 0, 0])
                load_smem_b(0, b, ~smem_b[0, 0, 0])
                cp_async_wait_all()

                syncthreads()
                for k0 in range((k_part_extent + block_k - 1) // block_k):
                    load_smem_a(k0 + 1, a, ~smem_a[(k0 + 1) % 2, 0, 0])
                    load_smem_b(k0 + 1, b, ~smem_b[(k0 + 1) % 2, 0, 0])
                    for mi in range(mma_count_m):
                        load_regs_a(mi, 0, ~smem_a[k0 % 2, 0, 0], ~regs_a[0, mi, 0])
                    for mj in range(mma_count_n):
                        load_regs_b(mj, 0, ~smem_b[k0 % 2, 0, 0], ~regs_b[0, mj, 0])
                    for mk in range(mma_count_k):
                        if mk + 1 < mma_count_k:
                            for mi in range(mma_count_m):
                                load_regs_a(mi, mk + 1, ~smem_a[k0 % 2, 0, 0], ~regs_a[(mk + 1) % 2, mi, 0])
                            for mj in range(mma_count_n):
                                load_regs_b(mj, mk + 1, ~smem_b[k0 % 2, 0, 0], ~regs_b[(mk + 1) % 2, mj, 0])
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            warp_mma(~regs_a[mk % 2, mi, 0], ~regs_b[mk % 2, mj, 0], ~regs_c[mi, mj, 0])
                    cp_async_wait_all()
                    syncthreads()

                # store back
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
                c_head_index = spatial(*c_head).map(blockIdx.z)
                gmem_c = c[c_head_index][offset_m:, offset_n:]

                if warp_count_k == 1:
                    for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            p = 0
                            for i, j in mma_config.c_store_map.on(lane_id):
                                delta_m = wi * warp_m + mi * mma_m + i
                                delta_n = wj * warp_n + mj * mma_n + j
                                in_bound = (offset_m + delta_m < m_size) and (offset_n + delta_n < n_size)
                                if in_bound:
                                    gmem_c[delta_m, delta_n] = regs_c[mi, mj, p]
                                p += 1
                else:
                    smem_c = tensor_pointer('float16', shape=[block_m, block_n])
                    smem_c = dynamic_shared_memory(byte_offset=0, dtype=float16)

                    for k_round in range(warp_count_k):
                        for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                            if wk == k_round:
                                for mi, mj in grid(mma_count_m, mma_count_n):
                                    p = 0
                                    for i, j in mma_config.c_store_map.on(lane_id):
                                        delta_m = wi * warp_m + mi * mma_m + i
                                        delta_n = wj * warp_n + mj * mma_n + j
                                        in_bound = (offset_m + delta_m < m_size) and (offset_n + delta_n < n_size)
                                        if in_bound:
                                            if k_round == 0:
                                                smem_c[delta_m, delta_n] = regs_c[mi, mj, p]
                                            else:
                                                smem_c[delta_m, delta_n] += regs_c[mi, mj, p]
                                        p += 1
                        if warp_count_k > 1:
                            syncthreads()
                    for i, j in store_smem_c_map.on(threadIdx.x):
                        if offset_m + i < m_size and offset_n + j < n_size:
                            gmem_c[i, j] = smem_c[i, j]

        ir_module = module.ir_module()
        assert isinstance(matmul_f16_kernel, Function)

        return ir_module


class MatmulF16Op(Operator):
    def __init__(self, a: Tensor, b: Tensor, parallel_k_parts=1):
        if not (isinstance(parallel_k_parts, int) and not isinstance(parallel_k_parts, bool)):
            raise ValueError('parallel_k_parts must be an integer, got {}'.format(parallel_k_parts))
        super().__init__(
            inputs=[a, b],
            attributes={'parallel_k_parts': parallel_k_parts},
            task=MatmulF16Task(input_like(a, 'a'), input_like(b, 'b'), parallel_k_parts),
        )


def matmul_f16(a: Tensor, b: Tensor, parallel_k_parts=1) -> Tensor:
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError('a and b must have at least 2 dimensions, got shape {} and {}'.format(a.shape, b.shape))
    # TODO: impliment dynamic run-time shape assertion
    if not (isinstance(a.shape[-1], Expr) or isinstance(b.shape[-1], Expr)) and (
        a.shape[-1] % 2 != 0 or b.shape[-1] % 2 != 0
    ):
        raise ValueError('Expect the last dimension of the input tensors to be a multiple of 2')
    if a.dtype != dtypes.float16 or b.dtype != dtypes.float16:
        raise ValueError('BatchMatmulF16Op only support float16, got {} and {}'.format(a.dtype, b.dtype))
    return MatmulF16Op(a, b, parallel_k_parts).outputs[0]

# %%

if __name__ == '__main__':
    import hidet
    from hidet.ir.type import tensor_type
    from hidet.lang import attrs, f16, i32, tensor_pointer
    from hidet.lang.layout import row_major
    from hidet.lang.cuda import dynamic_shared_memory, threadIdx

    asym = hidet.symbol_var('a', 'int32')
    asym = asym * 2 + asym % 2
    test_layout = row_major(64, asym)

    smem_a_type = tensor_type(
            'float16', shape=[64, asym], layout=test_layout
        )

    with hidet.script_module() as module:
        @hidet.script
        def test_fn(a: smem_a_type, i: i32):
            global smem_a_type
            smem_a_type = tensor_type(
                'float16', shape=[i, asym], layout=row_major(i, asym)
            )

            a[threadIdx.x, threadIdx.y] += 1
        
        @hidet.script
        def main(a: ~smem_a_type):
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 32
            # the second 2 means '2 bytes per float16'
            attrs.cuda.dynamic_smem_bytes = 64 * asym
            smem_a = tensor_pointer(
                'float16', shape=[64, asym], layout=test_layout
            )
            smem_a = dynamic_shared_memory(byte_offset=0, dtype=float16)
            test_fn(smem_a, threadIdx.x)

    ir_module = module.ir_module()
    # print(ir_module)
    tst = ir_module.build()
    tst.functions