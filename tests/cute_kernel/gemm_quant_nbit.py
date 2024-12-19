from typing import List
import hidet

from hidet.lang import attrs
from hidet.lang.cuda import (
    blockIdx,
    threadIdx,
    syncthreads,
    cp_async_wait_all,
    cp_async_commit_group,
    cp_async_wait_group,
)
from hidet.ir.primitives.cuda.mutex import acquire_seq_semaphore
from hidet.ir.primitives.cuda.atomic import atomic_add
from hidet.ir.expr import var

from hidet.ir.cute.layout import TensorLayout, make_layout, layout_auto
from hidet.ir.cute.layout import Level
from hidet.ir.cute.algorithm import MmaAtom, TiledMma, auto_copy
from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    mask,
    copy,
    mma,
    rearrange,
    cast,
    fill,
)

from hidet.ir.cute import auto_layout
from hidet.ir.cute import composition, coalesce

from hidet.utils.py import cdiv
from hidet.utils import initialize

from hidet.lang.types import i32, f16, u1, u2, u4
from hidet.ir.type import DataType

from hidet.ir.library import tune

from quant_utils import gemm_quant_module, weight_quantization_subbyte, weight_dequantization_subbyte, canonicalize


_predefined_tiled_mma: List[TiledMma] = []


@initialize()
def register_tiled_mma():
    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 2), TensorLayout((1, 2)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (2, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (2, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (2, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 2))
    warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (1, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)


class FpAIntBGemm:
    def __init__(self, m: int, k: int, n: int, group_size: int, bit_width: int = 3):
        self.m = m
        self.k = k
        self.n = n
        self.group_size = group_size
        assert bit_width in (1, 2, 3)
        self.bit_width = bit_width

    def deduce_mem_layout(self, tiled_mma: TiledMma, block_k: int, stages: int, wdtype: DataType):
        b_shape, b_tv_layout = tiled_mma.b_tv_layout()
        b_t, b_v = canonicalize(b_tv_layout)

        from hidet.transforms.cute.cuda.instruction_selection import memory_instructions
        from hidet.ir.cute.layout import group, left_inverse, right_inverse, prefix_product, filter, complement

        candidates = []
        for inst in memory_instructions:
            if inst.src_scope.is_shared() and inst.dst_scope.is_register():
                candidates.append(inst)

        cands = []
        for inst in candidates:
            dummy = var("x", ~wdtype)
            src_inst = inst.get_layout_in_element(dummy, inst.src_layout)
            dst_inst = inst.get_layout_in_element(dummy, inst.dst_layout)
            dst_thr_inst, dst_val_inst = dst_inst[0], dst_inst[1]
            if b_t.size() < dst_thr_inst.size():
                continue
            if b_v.size() < dst_val_inst.size():
                continue
            thr_inst, thr_rest = group(b_t, dst_thr_inst.size())
            val_inst, val_rest = group(b_v, dst_val_inst.size())
            cvt = coalesce(composition(make_layout(thr_inst, val_inst), left_inverse(dst_inst)))
            result_tv = composition(cvt, src_inst)
            result_thr, result_val = result_tv
            result_thr = coalesce(make_layout(result_thr, thr_rest))
            result_val = coalesce(make_layout(result_val, val_rest))
            result_tv = make_layout(result_thr, result_val)

            result_thr = filter(result_thr)
            result_val = filter(result_val)
            last_dim = result_val.size()
            shape = result_thr.shape_tuple
            stride = prefix_product(shape, last_dim)

            shape += (last_dim,)
            stride += (1,)
            mem = TensorLayout(shape, stride)
            crd2addr = coalesce(composition(mem, left_inverse(filter(result_tv))))
            m_mode, n_mode = group(crd2addr, b_shape[0])
            block = make_layout(m_mode, n_mode)
            n_shape = n_mode.shape + (block_k // n_mode.size(),)
            n_stride = n_mode.stride + (block.cosize(),)
            n_mode_ = TensorLayout(n_shape, n_stride)
            smem_layout = make_layout(m_mode, n_mode_)
            if stages > 1:
                stage_layout = TensorLayout(stages, smem_layout.cosize())
                smem_layout = make_layout(m_mode, n_mode_, stage_layout)

            n_shape = n_mode.shape + (self.k // n_mode.size(),)
            n_stride = n_mode.stride + (block.cosize(),)
            n_mode_ = TensorLayout(n_shape, n_stride)
            gmem_layout = make_layout(m_mode, n_mode_)

            cands.append((inst, smem_layout, gmem_layout))

        inst, smem_layout, gmem_layout = cands[-1]
        return smem_layout, gmem_layout

    def get_gmem(self, gmem_layout):
        m_mode, n_mode = gmem_layout
        m_shape = m_mode.shape + (self.n // m_mode.size(),)
        m_stride = m_mode.stride + (gmem_layout.cosize(),)
        m_mode = TensorLayout(m_shape, m_stride)
        return make_layout(n_mode, m_mode)

    def modules(self):
        return tune.extract_ir_modules(self._candidates)

    @tune.space(
        2,
        tiled_mma=_predefined_tiled_mma,
        block_k=[64, 128, 256],
        multi_buffer=[True, False],
        parallel_k_parts=[1, 2, 3, 4, 8, 16],
    )
    def _candidates(self, tiled_mma, block_k, multi_buffer, parallel_k_parts):
        if multi_buffer:
            return self.multi_buffer_kernel(tiled_mma, block_k, parallel_k_parts)
        else:
            return self.single_buffer_kernel(tiled_mma, block_k, parallel_k_parts)

    def _k_partition(self, tiled_mma: TiledMma, block_k: int, parallel_k_parts: int):
        k = self.k
        if parallel_k_parts == 1:
            return k

        k_partition = block_k
        while k_partition * parallel_k_parts < k:
            k_partition += block_k
        return k_partition

    def single_buffer_kernel(self, tiled_mma: TiledMma, block_k: int, parallel_k_parts: int):
        m, n, k = self.m, self.n, self.k
        group_size = self.group_size
        a_shape, a_tv_layout = tiled_mma.a_tv_layout()
        b_shape, b_tv_layout = tiled_mma.b_tv_layout()
        c_shape, c_tv_layout = tiled_mma.c_tv_layout()

        block_m, inst_k = a_shape
        block_n, inst_k_ = b_shape
        block_m_, block_n_ = c_shape
        assert block_m == block_m_ and block_n == block_n_ and inst_k == inst_k_

        a_t, a_v = canonicalize(a_tv_layout)
        b_t, b_v = canonicalize(b_tv_layout)
        c_t, c_v = canonicalize(c_tv_layout)

        threads = c_t.size()

        tune.check(m % block_m == 0)
        dynamic_smem_bytes = (
            block_m * block_k * f16.nbytes + block_n * block_k * u2.nbits // 8 + block_n * block_k * u1.nbits // 8
        )
        stages = 1

        smem_layout_2b, gmem_layout_2b = self.deduce_mem_layout(tiled_mma, block_k, stages, u2)
        smem_layout_1b, gmem_layout_1b = self.deduce_mem_layout(tiled_mma, block_k, stages, u1)

        gmem_2b = self.get_gmem(gmem_layout_2b)
        gmem_1b = self.get_gmem(gmem_layout_1b)

        qmod_2b = weight_quantization_subbyte(k, n, gmem_2b, u2)
        dqmod_2b = weight_dequantization_subbyte(k, n, gmem_2b, u2)

        qmod_1b = weight_quantization_subbyte(k, n, gmem_1b, u1)
        dqmod_1b = weight_dequantization_subbyte(k, n, gmem_1b, u1)

        k_partition = self._k_partition(tiled_mma, block_k, parallel_k_parts)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: f16[m, k],
                b2: u2[n, k],
                b1: u1[n, k],
                c: f16[m, n],
                scale: f16[k // group_size, n],
                bias: f16[k // group_size, n],
                c_parallel_k_parts: f16[parallel_k_parts, m, n],
                lock: i32[cdiv(m, block_m), cdiv(n, block_n)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(m, block_m) * cdiv(n, block_n), parallel_k_parts
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(m, block_m)
                num_pid_n = cdiv(n, block_n)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_part = blockIdx.y
                if k_part == 0 and threadIdx.x == 0:
                    lock[pid_m, pid_n] = 0
                k_start_pos = k_part * k_partition
                k_start_ofs2 = gmem_layout_2b((0, k_start_pos))
                k_start_ofs1 = gmem_layout_1b((0, k_start_pos))

                tr_a = make_tensor("float16", layout_auto((block_m, inst_k * 2)), "register")
                tr_b2 = make_tensor(u2, layout_auto((block_n, inst_k * 2)), "register")
                tr_b1 = make_tensor(u1, layout_auto((block_n, inst_k * 2)), "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                ts_a = make_tensor("float16", TensorLayout((block_m, block_k), (block_k, 1)), "shared")
                ts_b2 = make_tensor(u2, smem_layout_2b, "shared")
                ts_b1 = make_tensor(u1, smem_layout_1b, "shared")

                tg_a = tensor_view(a[pid_m * block_m :, k_start_pos:], TensorLayout((block_m, k), (k, 1)), "global")
                tg_b2 = tensor_view(b2[pid_n * block_n :, k_start_ofs2:], gmem_layout_2b, "global")
                tg_b1 = tensor_view(b1[pid_n * block_n :, k_start_ofs1:], gmem_layout_1b, "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb2 = partition_src(tg_b2, auto_copy())
                txsb2 = partition_dst(ts_b2, auto_copy())
                txgb1 = partition_src(tg_b1, auto_copy())
                txsb1 = partition_dst(ts_b1, auto_copy())

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb2 = partition_src(ts_b2, auto_copy())
                txrb2 = partition_dst(tr_b2, auto_copy())
                txSb1 = partition_src(ts_b1, auto_copy())
                txrb1 = partition_dst(tr_b1, auto_copy())

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + block_k - 1) // block_k
                k_tile_max = block_k // inst_k
                for ko in range(k_block_max):
                    copy(auto_copy((block_m, block_k)), txga[:, :, ko], txsa)
                    copy(auto_copy((block_n, block_k)), txgb2[:, :, ko], txsb2)
                    copy(auto_copy((block_n, block_k)), txgb1[:, :, ko], txsb1)
                    cp_async_wait_all()
                    syncthreads()

                    copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])
                    copy(auto_copy(), txSb2[:, :, 0], txrb2[:, :, 0])
                    copy(auto_copy(), txSb1[:, :, 0], txrb1[:, :, 0])

                    for ki in range(k_tile_max):
                        if ki < k_tile_max - 1:
                            copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSb2[:, :, ki + 1], txrb2[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSb1[:, :, ki + 1], txrb1[:, :, (ki + 1) % 2])

                        txrb2_f16 = cast(txrb2[:, :, ki % 2], f16)
                        txrb1_f16 = cast(txrb1[:, :, ki % 2], f16)
                        if self.bit_width == 3:
                            mma(tiled_mma, tr_c, txra[:, :, ki % 2], f16(2.0) * txrb2_f16 + txrb1_f16, tr_c)
                        elif self.bit_width == 2:
                            mma(tiled_mma, tr_c, txra[:, :, ki % 2], f16(2.0) * txrb2_f16, tr_c)
                        else:
                            mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb1_f16, tr_c)
                    syncthreads()

                tr_c_f16 = cast(tr_c, f16)
                tr_C = rearrange(tr_c_f16, auto_layout, "register")
                msk_c = mask(auto_copy(), [m - pid_m * block_m, n - pid_n * block_n])

                k_part = blockIdx.y
                lc = ~lock[pid_m, pid_n]
                if k_part < parallel_k_parts - 1:

                    tg_c = tensor_view(
                        c_parallel_k_parts[
                            k_part, pid_m * block_m : (pid_m + 1) * block_m, pid_n * block_n : (pid_n + 1) * block_n
                        ],
                        TensorLayout((block_m, block_n), (n, 1)),
                        "global",
                    )

                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((block_m, block_n)), txrx_c, txgx_c, msk_c)

                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lc, 1)
                else:
                    tr_c_k_part = make_tensor("float16", auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[
                                i, pid_m * block_m : (pid_m + 1) * block_m, pid_n * block_n : (pid_n + 1) * block_n
                            ],
                            TensorLayout((block_m, block_n), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((block_m, block_n)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * block_m : (pid_m + 1) * block_m, pid_n * block_n : (pid_n + 1) * block_n],
                        TensorLayout((block_m, block_n), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    txrx_c_final = partition_src(tr_C, auto_copy())
                    copy(auto_copy((block_m, block_n)), txrx_c_final, txgx_c_final, msk_c)

        return gemm_quant_module(script_module.ir_module(), (qmod_2b, qmod_1b), (dqmod_2b, dqmod_1b))

    def multi_buffer_kernel(self, tiled_mma: TiledMma, block_k: int, parallel_k_parts: int):
        m, n, k = self.m, self.n, self.k
        group_size = self.group_size
        a_shape, a_tv_layout = tiled_mma.a_tv_layout()
        b_shape, b_tv_layout = tiled_mma.b_tv_layout()
        c_shape, c_tv_layout = tiled_mma.c_tv_layout()

        block_m, inst_k = a_shape
        block_n, inst_k_ = b_shape
        block_m_, block_n_ = c_shape
        assert block_m == block_m_ and block_n == block_n_ and inst_k == inst_k_

        a_t, a_v = canonicalize(a_tv_layout)
        b_t, b_v = canonicalize(b_tv_layout)
        c_t, c_v = canonicalize(c_tv_layout)

        threads = c_t.size()

        dynamic_smem_bytes = (
            block_m * block_k * f16.nbytes + block_n * block_k * u2.nbits // 8 + block_n * block_k * u1.nbits // 8
        )

        compute_capability = hidet.option.cuda.get_arch_pair()
        compute_capability = compute_capability[0] * 10 + compute_capability[1]
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        max_smem = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        stages = max_smem // dynamic_smem_bytes
        tune.check(2 <= stages < 10)
        tune.check(m % block_m == 0)
        dynamic_smem_bytes *= stages

        smem_layout_2b, gmem_layout_2b = self.deduce_mem_layout(tiled_mma, block_k, stages, u2)
        smem_layout_1b, gmem_layout_1b = self.deduce_mem_layout(tiled_mma, block_k, stages, u1)

        gmem_2b = self.get_gmem(gmem_layout_2b)
        gmem_1b = self.get_gmem(gmem_layout_1b)

        qmod_2b = weight_quantization_subbyte(k, n, gmem_2b, u2)
        dqmod_2b = weight_dequantization_subbyte(k, n, gmem_2b, u2)

        qmod_1b = weight_quantization_subbyte(k, n, gmem_1b, u1)
        dqmod_1b = weight_dequantization_subbyte(k, n, gmem_1b, u1)

        k_partition = self._k_partition(tiled_mma, block_k, parallel_k_parts)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: f16[m, k],
                b2: u2[n, k],
                b1: u1[n, k],
                c: f16[m, n],
                scale: f16[k // group_size, n],
                bias: f16[k // group_size, n],
                c_parallel_k_parts: f16[parallel_k_parts, m, n],
                lock: i32[cdiv(m, block_m), cdiv(n, block_n)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(m, block_m) * cdiv(n, block_n), parallel_k_parts
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(m, block_m)
                num_pid_n = cdiv(n, block_n)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_part = blockIdx.y
                if k_part == 0 and threadIdx.x == 0:
                    lock[pid_m, pid_n] = 0
                k_start_pos = k_part * k_partition
                k_start_ofs2 = gmem_layout_2b((0, k_start_pos))
                k_start_ofs1 = gmem_layout_1b((0, k_start_pos))

                tr_a = make_tensor("float16", layout_auto((block_m, inst_k * 2)), "register")
                tr_b2 = make_tensor(u2, layout_auto((block_n, inst_k * 2)), "register")
                tr_b1 = make_tensor(u1, layout_auto((block_n, inst_k * 2)), "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                ts_a = make_tensor(
                    "float16", TensorLayout((block_m, block_k, stages), (block_k, 1, block_m * block_k)), "shared"
                )
                ts_b2 = make_tensor(u2, smem_layout_2b, "shared")
                ts_b1 = make_tensor(u1, smem_layout_1b, "shared")

                tg_a = tensor_view(a[pid_m * block_m :, k_start_pos:], TensorLayout((block_m, k), (k, 1)), "global")
                tg_b2 = tensor_view(b2[pid_n * block_n :, k_start_ofs2:], gmem_layout_2b, "global")
                tg_b1 = tensor_view(b1[pid_n * block_n :, k_start_ofs1:], gmem_layout_1b, "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb2 = partition_src(tg_b2, auto_copy())
                txsb2 = partition_dst(ts_b2, auto_copy())
                txgb1 = partition_src(tg_b1, auto_copy())
                txsb1 = partition_dst(ts_b1, auto_copy())

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + block_k - 1) // block_k
                for s in range(stages - 1):
                    if s < k_block_max:
                        copy(auto_copy((block_m, block_k)), txga[:, :, s], txsa[:, :, s])
                        copy(auto_copy((block_n, block_k)), txgb2[:, :, s], txsb2[:, :, s])
                        copy(auto_copy((block_n, block_k)), txgb1[:, :, s], txsb1[:, :, s])
                    cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()

                smem_pipe_read = 0
                smem_pipe_write = stages - 1

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb2 = partition_src(ts_b2, auto_copy())
                txrb2 = partition_dst(tr_b2, auto_copy())
                txSb1 = partition_src(ts_b1, auto_copy())
                txrb1 = partition_dst(tr_b1, auto_copy())

                txSa_p = txSa[:, :, :, smem_pipe_read]
                txSb2_p = txSb2[:, :, :, smem_pipe_read]
                txSb1_p = txSb1[:, :, :, smem_pipe_read]

                copy(auto_copy(), txSa_p[:, :, 0], txra[:, :, 0])
                copy(auto_copy(), txSb2_p[:, :, 0], txrb2[:, :, 0])
                copy(auto_copy(), txSb1_p[:, :, 0], txrb1[:, :, 0])

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + block_k - 1) // block_k
                k_tile_max = block_k // inst_k
                for ko in range(k_block_max):
                    for ki in range(k_tile_max):
                        if ki == k_tile_max - 1:
                            cp_async_wait_group(allow_on_fly_groups=stages - 2)
                            syncthreads()

                        k_tile_next = (ki + 1) % k_tile_max
                        copy(auto_copy(), txSa[:, :, k_tile_next, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSb2[:, :, k_tile_next, smem_pipe_read], txrb2[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSb1[:, :, k_tile_next, smem_pipe_read], txrb1[:, :, (ki + 1) % 2])
                        if ki == 0:
                            if ko + stages - 1 < k_block_max:
                                copy(
                                    auto_copy((block_m, block_k)),
                                    txga[:, :, ko + stages - 1],
                                    txsa[:, :, smem_pipe_write],
                                )
                                copy(
                                    auto_copy((block_n, block_k)),
                                    txgb2[:, :, ko + stages - 1],
                                    txsb2[:, :, smem_pipe_write],
                                )
                                copy(
                                    auto_copy((block_n, block_k)),
                                    txgb1[:, :, ko + stages - 1],
                                    txsb1[:, :, smem_pipe_write],
                                )
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        if ki == k_tile_max - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                        txrb2_f16 = cast(txrb2[:, :, ki % 2], f16)
                        txrb1_f16 = cast(txrb1[:, :, ki % 2], f16)
                        if self.bit_width == 3:
                            mma(tiled_mma, tr_c, txra[:, :, ki % 2], f16(2.0) * txrb2_f16 + txrb1_f16, tr_c)
                        elif self.bit_width == 2:
                            mma(tiled_mma, tr_c, txra[:, :, ki % 2], f16(2.0) * txrb2_f16, tr_c)
                        else:
                            mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb1_f16, tr_c)

                tr_c_f16 = cast(tr_c, f16)
                tr_C = rearrange(tr_c_f16, auto_layout, "register")
                msk_c = mask(auto_copy(), [m - pid_m * block_m, n - pid_n * block_n])

                k_part = blockIdx.y
                lc = ~lock[pid_m, pid_n]
                if k_part < parallel_k_parts - 1:

                    tg_c = tensor_view(
                        c_parallel_k_parts[
                            k_part, pid_m * block_m : (pid_m + 1) * block_m, pid_n * block_n : (pid_n + 1) * block_n
                        ],
                        TensorLayout((block_m, block_n), (n, 1)),
                        "global",
                    )

                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((block_m, block_n)), txrx_c, txgx_c, msk_c)

                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lc, 1)
                else:
                    tr_c_k_part = make_tensor("float16", auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[
                                i, pid_m * block_m : (pid_m + 1) * block_m, pid_n * block_n : (pid_n + 1) * block_n
                            ],
                            TensorLayout((block_m, block_n), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((block_m, block_n)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * block_m : (pid_m + 1) * block_m, pid_n * block_n : (pid_n + 1) * block_n],
                        TensorLayout((block_m, block_n), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    txrx_c_final = partition_src(tr_C, auto_copy())
                    copy(auto_copy((block_m, block_n)), txrx_c_final, txgx_c_final, msk_c)

        return gemm_quant_module(script_module.ir_module(), (qmod_2b, qmod_1b), (dqmod_2b, dqmod_1b))
