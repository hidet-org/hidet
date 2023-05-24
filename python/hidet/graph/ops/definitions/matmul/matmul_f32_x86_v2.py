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
from typing import List, Union, Tuple
from hidet.ir.dtypes import float32, int32, boolean
from hidet.ir.expr import cast
from hidet.ir.func import IRModule, Function
from hidet.ir.compute import TensorNode
from hidet.ir.stmt import DeclareScope
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.definitions.utils import input_like, broadcast_shape, can_mutually_broadcast
from hidet.graph.ops.definitions.utils import tune
from hidet.graph.operator import Operator, Tensor
from hidet.graph.ops.definitions.utils import broadcast_indices
from hidet.ir.primitives import sqrt
from hidet.ir.type import void_p


class MatmulF32Taskx86OneDNN(Task):
    def __init__(self, a: TensorNode, b: TensorNode):
        a_shape = a.const_shape
        b_shape = b.const_shape

        if not a.type.dtype == float32 or not b.type.dtype == float32:
            raise ValueError('Both inputs must be float32 tensors')

        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError('Matrix multiplication expect at least 2D tensor, got {} and {}'.format(a_shape, b_shape))

        if a_shape[-1] != b_shape[-2]:
            raise ValueError(
                'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                ', got {} and {}'.format(a_shape, b_shape)
            )
        k_size = a_shape[-1]
        c_shape = broadcast_shape(a_shape[:-2], b_shape[:-2]) + [a_shape[-2], b_shape[-1]]

        c = compute(
            name='c',
            shape=c_shape,
            fcompute=lambda *indices: reduce(
                shape=[k_size],
                fcompute=lambda k: a[broadcast_indices(indices[:-2], a_shape[:-2], c_shape[1:-2]) + [indices[-2], k]]
                                   * b[broadcast_indices(indices[:-2], b_shape[:-2], c_shape[1:-2]) + [k, indices[-1]]],
                reduce_type='sum'
            )
        )

        super().__init__(
            name='matmul_f32_x86', inputs=[a, b], outputs=[c], attributes={
                'm_size': a_shape[-2],
                'n_size': b_shape[-1],
                'k_size': a_shape[-1]
            }
        )

    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_matmulf32_x86)

    @tune.space(0, 'block_m', [4032])
    @tune.space(0, 'block_n', [96])
    @tune.space(0, 'block_k', [96])
    @tune.space(0, 'nthrs', [16])
    @tune.space(0, 'micro_ker', [(6, 16)])
    def schedule_matmulf32_x86(self, block_m=4032, block_n=96, block_k=96, nthrs=32, micro_ker=(6, 16)):
        import hidet
        from hidet.ir.type import tensor_type
        from hidet.lang import tensor, grid, as_tensor_pointer
        from hidet.lang.layout import row_layout, col_layout
        from hidet.lang.avx import avx_f32x8_store, avx_f32x8_fmadd, avx_f32x8_load, avx_f32x8_broadcast
        from hidet.lang.avx import aligned_alloc

        node_a, node_b, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape: Tuple[int] = node_a.const_shape
        b_shape: Tuple[int] = node_b.const_shape
        c_shape: Tuple[int] = node_c.const_shape
        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]

        tile_m, tile_n = micro_ker
        tune.check(block_m % tile_m == block_n % tile_n == 0, 'Tile size must divide the corresponding block size')

        # TODO: Do I still want to pack it? If so add variables here
        DTYPE_SIZE = 4
        PAGE_4K = 4096

        with hidet.script_module() as module:
            HIDET_NULL = int32(0)
            HIDET_NULLPTR = ~HIDET_NULL

            @hidet.script
            def div_up(a: int32, b: int32):
                assert b != 0, "division by 0"
                return (a + b - 1) // b

            @hidet.script
            def rnd_up(a: int32, b: int32):
                return div_up(a, b) * b

            @hidet.script
            def rnd_dn(a: int32, b: int32):
                return (a // b) * b

            @hidet.script
            def calc_nthr_nocopy_avx():
                BM_NOCOPY_AVX = 64
                BN_NOCOPY_AVX = 48
                BK_NOCOPY_AVX = 384
                BN_LARGE_NOCOPY_AVX = 192
                BM_SMALL_NOCOPY_AVX = 16
                BN_SMALL_NOCOPY_AVX = 1
                BK_SMALL_NOCOPY_AVX = 4

                nthr = nthrs
                nthr_m = (m_size + BM_NOCOPY_AVX - 1) // BM_NOCOPY_AVX
                nthr_n = (n_size + BN_NOCOPY_AVX - 1) // BN_NOCOPY_AVX
                nthr_k = 1

                # Partitioning along K dimension
                # TODO: The ref_gemm.cpp checks dnnl_thr_syncable(), but we only use OpenMP for now
                nthr_other = nthr_k
                assert nthr_other == 1
                while nthr_m * nthr_n * nthr_other < nthr and \
                        k_size // (nthr_other + 1) > BK_NOCOPY_AVX:
                    nthr_other += 1
                    if (nthr // nthr_other) * nthr_other > 0.9 * nthr:
                        nthr_k = nthr_other

                nthr = nthr // nthr_k
                if nthr_m == 1:
                    nthr_n = nthr
                if nthr_n == 1:
                    nthr_m = nthr
                # Simple partition reduction
                while nthr_m * nthr_n > nthr:
                    if nthr_m > nthr_n:
                        nthr_m -= 1
                    else:
                        nthr_n -= 1
                while nthr_m * nthr_n < nthr:
                    if nthr_m * nthr_n < nthr:
                        if nthr_m < nthr_n:
                            nthr_m += 1
                        else:
                            nthr_n += 1
                if nthr_m * nthr_n > nthr and nthr_m > 1 and nthr_n > 1:
                    if nthr_m <= nthr_n:
                        nthr_m = int32(sqrt(float32(nthr)))
                        if nthr_m > (m_size + BM_SMALL_NOCOPY_AVX - 1) // BM_SMALL_NOCOPY_AVX:
                            nthr_m = (m_size + BM_SMALL_NOCOPY_AVX - 1) // BM_SMALL_NOCOPY_AVX
                        nthr_n = nthr // nthr_m

                        while nthr_m > 1 and nthr_m * nthr_n != nthr:
                            nthr_m -= 1
                            nthr_n = nthr // nthr_m
                    else:
                        nthr_n = int32(sqrt(float32(nthr)))
                        if nthr_n > (n_size + BN_SMALL_NOCOPY_AVX - 1) // BN_SMALL_NOCOPY_AVX:
                            nthr_n = (n_size + BN_SMALL_NOCOPY_AVX - 1) // BN_SMALL_NOCOPY_AVX
                        nthr_m = nthr // nthr_n

                        while nthr_n > 1 and nthr_m * nthr_n != nthr:
                            nthr_n -= 1
                            nthr_m = nthr // nthr_n

                MB = (m_size + nthr_m - 1) // nthr_m + BM_SMALL_NOCOPY_AVX - 1
                MB -= MB % BM_SMALL_NOCOPY_AVX
                NB = (n_size + nthr_n - 1) // nthr_n + BN_SMALL_NOCOPY_AVX - 1
                NB -= NB % BN_SMALL_NOCOPY_AVX
                KB = (k_size + nthr_k - 1) // nthr_k + BK_SMALL_NOCOPY_AVX - 1
                KB -= KB % BK_SMALL_NOCOPY_AVX

                if MB * nthr_m > m_size:
                    nthr_m = (m_size + MB - 1) // MB
                if NB * nthr_n > n_size:
                    nthr_n = (n_size + NB - 1) // NB
                if KB * nthr_k > k_size:
                    nthr_k = (k_size + KB - 1) // KB

                return nthr_m, nthr_n, nthr_k, MB, NB, KB

            @hidet.script
            def get_thr_block(NB: int32, N: int32, ithr: int32):
                start_pt = NB * ithr
                end_pt = start_pt + NB
                if end_pt > N:
                    end_pt = N
                myN = end_pt - start_pt

                return start_pt, end_pt, myN

            @hidet.script
            def kernel_6x16(K: int32, a_ptr: ~float32, b_ptr: ~float32, c_ptr: ~float32,
                            msize: int32, nsize: int32):
                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[msize, nsize])
                a = as_tensor_pointer(a_ptr, dtype=float32, shape=[m_size, k_size])
                b = as_tensor_pointer(b_ptr, dtype=float32, shape=[k_size, n_size])

                c0 = avx_f32x8_load(~c[0, 0])
                c08 = avx_f32x8_load(~c[0, 8])
                c1 = avx_f32x8_load(~c[1, 0])
                c18 = avx_f32x8_load(~c[1, 8])
                c2 = avx_f32x8_load(~c[2, 0])
                c28 = avx_f32x8_load(~c[2, 8])
                c3 = avx_f32x8_load(~c[3, 0])
                c38 = avx_f32x8_load(~c[3, 8])
                c4 = avx_f32x8_load(~c[4, 0])
                c48 = avx_f32x8_load(~c[4, 8])
                c5 = avx_f32x8_load(~c[5, 0])
                c58 = avx_f32x8_load(~c[5, 8])
                for k in range(K):
                    bb0to7 = avx_f32x8_load(~b[k, 0])
                    bb8to15 = avx_f32x8_load(~b[k, 8])

                    aa = avx_f32x8_broadcast(~a[0, k])
                    c0 = avx_f32x8_fmadd(aa, bb0to7, c0)
                    c08 = avx_f32x8_fmadd(aa, bb8to15, c08)
                    aa = avx_f32x8_broadcast(~a[1, k])
                    c1 = avx_f32x8_fmadd(aa, bb0to7, c1)
                    c18 = avx_f32x8_fmadd(aa, bb8to15, c18)
                    aa = avx_f32x8_broadcast(~a[2, k])
                    c2 = avx_f32x8_fmadd(aa, bb0to7, c2)
                    c28 = avx_f32x8_fmadd(aa, bb8to15, c28)
                    aa = avx_f32x8_broadcast(~a[3, k])
                    c3 = avx_f32x8_fmadd(aa, bb0to7, c3)
                    c38 = avx_f32x8_fmadd(aa, bb8to15, c38)
                    aa = avx_f32x8_load(~a[4, k])
                    c4 = avx_f32x8_fmadd(aa, bb0to7, c4)
                    c48 = avx_f32x8_fmadd(aa, bb8to15, c48)
                    aa = avx_f32x8_load(~a[5, k])
                    c5 = avx_f32x8_fmadd(aa, bb0to7, c5)
                    c58 = avx_f32x8_fmadd(aa, bb0to7, c58)
                avx_f32x8_store(~c[0, 0], c0)
                avx_f32x8_store(~c[0, 8], c08)
                avx_f32x8_store(~c[1, 0], c1)
                avx_f32x8_store(~c[1, 8], c18)
                avx_f32x8_store(~c[2, 0], c2)
                avx_f32x8_store(~c[2, 8], c28)
                avx_f32x8_store(~c[3, 0], c3)
                avx_f32x8_store(~c[3, 8], c38)
                avx_f32x8_store(~c[4, 0], c4)
                avx_f32x8_store(~c[4, 8], c48)
                avx_f32x8_store(~c[5, 0], c5)
                avx_f32x8_store(~c[5, 8], c58)

            @hidet.script
            def block_ker(M: int32, N: int32, K: int32,
                          a_ptr: ~float32, b_ptr: ~float32, c_ptr: ~float32,
                          ws: ~float32, do_copy: boolean):
                Nu = rnd_dn(N, tile_n)  # TODO: unroll_factor::n in oneDNN is this right...
                Mu = rnd_dn(M, tile_m)
                a = as_tensor_pointer(a_ptr, dtype=float32, shape=[m_size, k_size])
                b = as_tensor_pointer(b_ptr, dtype=float32, shape=[k_size, n_size])
                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[m_size, n_size])

                i = 0
                while i < Mu:
                    j = 0
                    while j < Nu:
                        cur_b = ~b[0, j]
                        cur_a = ~a[i, 0]
                        # if do_copy:
                        #     if j == 0:
                        #         for
                        # TODO: Figure out this 'do_copy' thing after getting the rest working
                        kernel_6x16(K, cur_a, cur_b, ~c[i, j])

                        j += tile_n
                    i += tile_m
                # Tail processing
                for ii in range(M):
                    for jj in range(N):
                        c_acc = c[ii, jj]
                        for kk in range(K):
                            c_acc += a[ii, kk] + b[kk, jj]
                        c[ii, jj] = c_acc
                # Tail processing continued
                ii = Mu
                while ii < M:
                    jj = Nu
                    while jj < N:
                        c_acc = c[ii, jj]
                        for kk in range(K):
                            c_acc += a[ii, kk] * b[kk, jj]
                        c[ii, jj] = c_acc
                        jj += 1
                    ii += 1

            @hidet.script
            def gemm_ithr(M: int32, N: int32, K: int32,
                          a_ptr: ~float32, b_ptr: ~float32, c_ptr: ~float32,
                          ws: ~float32, do_copy: boolean, cm: int32, cn: int32):
                # TODO: The 'BM/BN/BK' in oneDNN should be equal to the 'block_xx' here right...
                if M <= 0 and N <= 0:
                    return
                c = as_tensor_pointer(c_ptr, float32, shape=[cm, cn])
                a = as_tensor_pointer(a_ptr, float32, shape=[m_size, k_size])
                b = as_tensor_pointer(b_ptr, float32, shape=[k_size, n_size])
                if K <= 0:
                    return

                Bk = 0
                while Bk < K:
                    kb = min(K - Bk, block_k)
                    Bm = 0
                    while Bm < M:
                        mb = min(M - Bm, block_m)
                        Bn = 0
                        while Bn < N:
                            nb = min(N - Bn, block_n)
                            cur_a = ~a[Bm, Bk]
                            cur_b = ~b[Bk, Bn]
                            cur_c = ~c[Bm, Bn]
                            block_ker(mb, nb, kb, cur_a, cur_b, cur_c, ws, do_copy)
                            Bn += block_n
                        Bm += block_m
                    Bk += block_k

            @hidet.script
            def partition_unit_diff(ithr: int32, nthr: int32, n: int32):
                band = n // nthr
                if band == 0:
                    band = 1
                tail = n - band * nthr
                t_offset = -1
                t_block = -1
                if tail < 0:
                    tail = 0
                if ithr < tail:
                    band += 1
                    t_offset = band * ithr
                    t_block = band
                else:
                    t_offset = band * ithr + tail
                    t_block = band

                assert t_offset > -1 and t_block > -1
                if t_offset >= n:
                    t_offset = 0
                    t_block = 0
                if t_offset + t_block > n:
                    t_block = n - t_offset
                return t_offset, t_block

            @hidet.script
            def sum_two_matrices(m: int32, n: int32, p_src: ~float32,
                                 src_m: int32, src_n: int32,
                                 p_dst: ~float32, ):
                my_c = as_tensor_pointer(p_src, dtype=float32, shape=[src_m, src_n])
                c = as_tensor_pointer(p_dst, dtype=float32, shape=[m_size, n_size])

                for i in range(m):
                    for j in range(n):
                        c[i, j] += my_c[i, j]

            @hidet.script
            def matmul_kernel_onednn(
                    a_ptr: ~float32, b_ptr: ~float32, c_ptr: ~float32
            ):
                a = as_tensor_pointer(a_ptr, dtype=float32, shape=[m_size, k_size])
                b = as_tensor_pointer(b_ptr, dtype=float32, shape=[k_size, n_size])
                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[m_size, n_size])

                # nthr_m, nthr_n, nthr_k, MB, NB, KB = calc_nthr_nocopy_avx()
                return_tuple = calc_nthr_nocopy_avx()
                nthr_m = return_tuple[0]
                nthr_n = return_tuple[1]
                nthr_k = return_tuple[2]
                MB = return_tuple[3]
                NB = return_tuple[4]
                KB = return_tuple[5]
                c_buffers = cast(HIDET_NULLPTR, ~float32)
                ws_buffers = cast(HIDET_NULLPTR, ~float32)
                if nthr_k > 1:
                    c_buffers = aligned_alloc(PAGE_4K, DTYPE_SIZE * nthr_m * nthr_n * (nthr_k - 1) * MB * NB)
                if not c_buffers:
                    nthr_k = 1
                    KB = k_size
                # TODO: If things go wrong in the future, check if really k_size is the 'K' in oneDNN
                do_copy: bool = NB // tile_n > 3  # TODO: tile_n is the unroll_factor<data_t>::n?
                nthr_mn = nthr_m * nthr_n
                nthr_to_use = nthr_mn * nthr_k
                ws_elems_per_thr = k_size * tile_m
                ws_size_per_thr = rnd_up(ws_elems_per_thr * DTYPE_SIZE, PAGE_4K)

                if do_copy:
                    ws_buffers = aligned_alloc(PAGE_4K, nthr_to_use * ws_size_per_thr)
                    if not ws_buffers:
                        do_copy = False

                # Similar to the parallel(int, lambda) in oneDNN
                thread_attr = 'p' + str(nthr_to_use)
                for ithr in grid(nthr_to_use, attrs=thread_attr):
                    ithr_mn = ithr % nthr_mn
                    ithr_m = ithr_mn % nthr_m
                    ithr_n = ithr_mn // nthr_m
                    ithr_k = ithr // nthr_mn

                    cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1)

                    ws = cast(HIDET_NULLPTR, ~float32)
                    if do_copy:
                        ws = ~ws_buffers[ithr * ws_size_per_thr // DTYPE_SIZE]

                    m_from, m_to, myM = get_thr_block(MB, m_size, ithr_m)
                    n_from, n_to, myN = get_thr_block(NB, n_size, ithr_n)
                    k_from, k_to, myK = get_thr_block(KB, k_size, ithr_k)

                    if myM > 0 and myN > 0:
                        myC = ~c[m_from, n_from]
                        cm, cn = (m_size, n_size)
                        if ithr_k > 0:
                            myC = ~c_buffers[MB * NB * (cbase + ithr_k - 1)]
                            cm, cn = (MB, NB)

                        myA_ptr = ~a[m_from, k_from]
                        myB_ptr = ~b[k_from, n_from]

                        gemm_ithr(myM, myN, myK, myA_ptr, myB_ptr, myC, ws, do_copy, cm, cn)
                if nthr_k > 1:
                    for ithr in grid(nthr_to_use, attrs=thread_attr):
                        ithr_mn = ithr % nthr_mn
                        ithr_m = ithr_mn % nthr_m
                        ithr_k = ithr // nthr_mn
                        ithr_n = ithr_mn // nthr_m

                        cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1)

                        m_from, m_to, myM = get_thr_block(MB, m_size, ithr_m)
                        n_from, n_to, myN = get_thr_block(NB, n_size, ithr_n)

                        # sum matrices partitioned along K dimension
                        offset, block = partition_unit_diff(ithr_k, nthr_k, myN)
                        for ik in range(nthr_k):
                            myC = ~c_buffers[MB * (NB * (cbase + ik - 1) + offset)]
                            sum_two_matrices(myM, block, myC, src_m=MB, src_n=NB,
                                             p_dst=~c[m_from, n_from + offset])

        assert isinstance(matmul_kernel_onednn, hidet.ir.Function)
        matmul_kernel_onednn.kind = 'host_kernel'
        ir_module = module.ir_module()
        return ir_module


class MatmulX86OneDNNOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        if not (len(a.shape) == len(b.shape) == 2 and a.shape[1] == b.shape[0]):
            raise ValueError(
                'Matrix multiplication: incompatible sizes: {} and {}'.format(
                    a.shape, b.shape
                )
            )
        task = MatmulF32Taskx86OneDNN(input_like(a, 'a'), input_like(b, 'b'))
        super().__init__(inputs=[a, b], attributes={}, task=task)


def matmul_x86_onednn(a: Tensor, b: Tensor) -> Tensor:
    return MatmulX86OneDNNOp(a, b).get_output(0)
