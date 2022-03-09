from typing import Mapping, Any

from hidet.implement.implementer import register_impl, Implementer, NotSupportedError
from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorCompute, ReduceCompute
from hidet.ir.dialects.pattern import TaskPattern, any_const_ints, any_scalar_expr, int_vars
from hidet.ir.expr import Var, Constant, And, Or
from hidet.ir.functors import rewrite
from hidet.ir.layout import TaskLayout, DataLayout
from hidet.ir.node import Node
from hidet.ir.primitives import block_idx, thread_idx, syncthreads, printf
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.task import Task, Grid
from hidet.ir.type import TensorType
from hidet.utils import prod


class Schedule:
    def __init__(self, block_warps=(4, 2), warp_outer=(2, 2), warp_middle=(4, 8), warp_inner=(4, 4), block_k=8):
        # configs
        self.block_warps = block_warps
        self.warp_outer = warp_outer
        self.warp_middle = warp_middle
        self.warp_inner = warp_inner
        self.block_k = block_k

        # sanity check
        block_threads = prod(block_warps) * 32
        block_shape = [prod(s) for s in zip(block_warps, warp_outer, warp_middle, warp_inner)]
        self.check(block_threads % block_k == 0, 'for a_g2s/b_g2s')
        self.check(block_shape[0] % (block_threads // block_k) == 0, 'for a_g2s')
        self.check(block_shape[1] % (block_threads // block_k) == 0, 'for b_g2s')

        # derived task layouts
        row_major = TaskLayout.row_major
        full = TaskLayout.full_layout
        self.a_g2s_layout = row_major([block_threads // block_k, block_k]) * full([block_shape[0] // (block_threads // block_k), 1])
        self.b_g2s_layout = full([1, block_shape[1] // (block_threads // block_k)]) * row_major([block_k, block_threads // block_k])
        self.a_s2r_layout = (row_major(block_warps) * full([warp_outer[0], 1]) * row_major(warp_middle) * full([warp_inner[0], 1])).projection({1: 0})
        self.b_s2r_layout = (row_major(block_warps) * full([1, warp_outer[1]]) * row_major(warp_middle) * full([1, warp_inner[1]])).projection({0: 0})
        self.c_mma_layout = row_major(block_warps) * full(warp_outer) * row_major(warp_middle) * full(warp_inner)
        self.c_r2g_layout = self.c_mma_layout

        # derived data layouts
        row_major = DataLayout.row_major
        column_major = DataLayout.column_major
        local = DataLayout.local
        self.smem_a_layout = column_major([block_shape[0], block_k])
        self.smem_b_layout = row_major([block_k, block_shape[1]])
        self.regs_a_layout = local([block_warps[0], 1]) * row_major([warp_outer[0], 1]) * local([warp_middle[0], 1]) * row_major([warp_inner[0], 1])
        self.regs_b_layout = local([1, block_shape[1]]) * row_major([1, warp_outer[1]]) * local([1, warp_middle[1]]) * row_major([1, warp_inner[1]])
        self.regs_c_layout = local(block_warps) * row_major(warp_outer) * local(warp_middle) * row_major(warp_inner)

        # derived constants
        self.block_threads = block_threads
        self.block_shape = block_shape

    def check(self, cond, msg=None):
        if not cond:
            raise NotSupportedError(msg)


class Pattern:
    def __init__(self):
        self.n, self.c, self.p, self.q, self.rc, self.rx, self.ry = any_const_ints(7)
        self.an, self.ac, self.ap, self.aq, self.arc, self.arx, self.ary = int_vars(['an', 'ac', 'ap', 'aq', 'arc', 'arx', 'ary'])
        self.x_expr = any_scalar_expr(exclude_vars=[self.ac])  # can only use [n, rc, p, q, rx, ry]
        self.w_expr = any_scalar_expr(exclude_vars=[self.an, self.ap, self.aq])  # can only use [c, rc, rx, ry]
        self.task_pattern = TaskPattern(
            compute_pattern=TensorCompute(
                name='out',
                shape=[self.n, self.c, self.p, self.q],
                axes=[self.an, self.ac, self.ap, self.aq],
                value=ReduceCompute(
                    shape=[self.rc, self.rx, self.ry],
                    axes=[self.arc, self.arx, self.ary],
                    value=self.x_expr * self.w_expr,
                    reduce_type='sum'
                )
            ),
            worker=Grid()
        )


def pattern2matched(pattern, match):
    matched = type(pattern)()
    for name in matched.__dict__:
        v = match[pattern.__dict__[name]]
        if isinstance(v, Constant):
            v = v.value
        matched.__dict__[name] = v
    return matched


@register_impl('cuda_grid_static_conv2d_implicit_gemm_implementer')
class CudaGridStaticConv2dImplicitGemmImplementer(Implementer):
    def __init__(self):
        self.pattern = Pattern()

    def priority(self) -> int:
        return 2

    def task_pattern(self) -> TaskPattern:
        return self.pattern.task_pattern

    def implement(self, task: Task, match: Mapping[Node, Any]) -> IRModule:
        default_schedule = Schedule()
        return self.implement_schedule(task, match, default_schedule)

    def implement_schedule(self, task: Task, match: Mapping[Node, Any], schedule: Schedule) -> IRModule:
        d: Pattern = pattern2matched(self.pattern, match)
        n, c, p, q, rc, rx, ry = d.n, d.c, d.p, d.q, d.rc, d.rx, d.ry
        block_shape, block_k = schedule.block_shape, schedule.block_k
        gemm_m = d.n * d.p * d.q
        gemm_n = d.c
        gemm_k = d.rc * d.rx * d.ry
        grid_layout = TaskLayout.row_major([(a + b - 1) // b for a, b in zip([gemm_m, gemm_n], block_shape)])
        worker = Grid(grid_dim=grid_layout.num_workers, block_dim=schedule.block_threads)

        with FunctionBuilder(f'{task.name}_grid', attrs={'worker': worker}) as fb:
            # params
            params = [Var(param.name, param_type) for param, param_type in zip(task.params, task.params_type)]
            fb.extend_params(params)
            param_map = {task_param: func_param for task_param, func_param in zip(task.params, params)}
            x_expr = rewrite(d.x_expr, param_map)
            w_expr = rewrite(d.w_expr, param_map)

            # local vars
            smem_a = Var(hint='smem_a', type=TensorType(scope='shared', dtype='float32', layout=schedule.smem_a_layout))
            smem_b = Var(hint='smem_b', type=TensorType(scope='shared', dtype='float32', layout=schedule.smem_b_layout))
            regs_a = Var(hint='regs_a', type=TensorType(scope='register', dtype='float32', layout=schedule.regs_a_layout))
            regs_b = Var(hint='regs_b', type=TensorType(scope='register', dtype='float32', layout=schedule.regs_b_layout))
            regs_c = Var(hint='regs_c', type=TensorType(scope='register', dtype='float32', layout=schedule.regs_c_layout))
            fb.extend_local_vars([smem_a, smem_b, regs_a, regs_b, regs_c])

            # body
            block_k_tiles = (gemm_k + block_k - 1) // block_k
            sb = StmtBuilder()
            # clear smem and accumulator
            # for ii, kk in schedule.a_g2s_layout(thread_idx()):
            #     sb += BufferStoreStmt(smem_a, [ii, kk], 0.0)
            # for kk, jj in schedule.b_g2s_layout(thread_idx()):
            #     sb += BufferStoreStmt(smem_b, [kk, jj], 0.0)
            for ii, jj in schedule.c_r2g_layout(thread_idx()):
                sb += BufferStoreStmt(regs_c, [ii, jj], 0.0)
            with sb.lets(['bi', 'bj'], grid_layout(block_idx())[0]) as (bi, bj):
                with sb.for_loop('bk', block_k_tiles) as bk:
                    # global memory -> shared memory
                    for ii, kk in schedule.a_g2s_layout(thread_idx()):
                        gi = ii + bi * block_shape[0]
                        gk = kk + bk * block_k
                        rmap = {
                            d.an: gi // (p * q),
                            d.ap: (gi // q) % p,
                            d.aq: gi % q,
                            d.arc: gk // (rx * ry),
                            d.arx: (gk // ry) % rx,
                            d.ary: gk % ry
                        }
                        with sb.if_then(And.join(gi < gemm_m, gk < gemm_k)):
                            sb += BufferStoreStmt(smem_a, [ii, kk], rewrite(x_expr, rmap))
                        with sb.otherwise():
                            sb += BufferStoreStmt(smem_a, [ii, kk], 0.0)
                    for kk, jj in schedule.b_g2s_layout(thread_idx()):
                        gk = kk + bk * block_k
                        gj = jj + bj * block_shape[1]
                        rmap = {
                            d.ac: gj,
                            d.arc: gk // (rx * ry),
                            d.arx: (gk // ry) % rx,
                            d.ary: gk % ry
                        }
                        with sb.if_then(And.join(gk < gemm_k, gj < gemm_n)):
                            sb += BufferStoreStmt(smem_b, [kk, jj], rewrite(w_expr, rmap))
                        with sb.otherwise():
                            sb += BufferStoreStmt(smem_b, [kk, jj], 0.0)
                    sb += syncthreads()
                    with sb.for_loop('wk', block_k) as wk:
                        # shared memory to local memory
                        for ii, kk in schedule.a_s2r_layout(thread_idx()):
                            sb += BufferStoreStmt(regs_a, [ii, kk], smem_a[ii, wk + kk])
                        for kk, jj in schedule.b_s2r_layout(thread_idx()):
                            sb += BufferStoreStmt(regs_b, [kk, jj], smem_b[wk + kk, jj])
                        # compute
                        for ii, jj in schedule.c_mma_layout(thread_idx()):
                            kk = 0
                            sb += BufferStoreStmt(regs_c, [ii, jj], regs_c[ii, jj] + regs_a[ii, kk] * regs_b[kk, jj])
                    sb += syncthreads()
                # c write back
                out = param_map[task.compute]
                for ii, jj in schedule.c_r2g_layout(thread_idx()):
                    gi = ii + bi * block_shape[0]
                    gj = jj + bj * block_shape[1]
                    with sb.if_then(And.join(gi < gemm_m, gj < gemm_n)):
                        an = gi // (p * q)
                        ac = gj
                        ap = (gi // q) % p
                        aq = gi % q
                        sb += BufferStoreStmt(out, [an, ac, ap, aq], regs_c[ii, jj])
            fb.set_body(sb.finish())

        func = fb.get()
        return IRModule(funcs={func.name: func}, task=task)
