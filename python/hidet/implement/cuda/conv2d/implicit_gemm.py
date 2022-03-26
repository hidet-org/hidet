import contextlib
from typing import Mapping, Any, List, Tuple, Union

from hidet.implement.implementer import register_impl, Implementer, NotSupportedError, Schedule
from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorCompute, ReduceCompute
from hidet.ir.dialects.pattern import TaskPattern, any_const_ints, any_scalar_expr, int_vars
from hidet.ir.expr import Var, Constant, And, if_then_else, Equal
from hidet.ir.stmt import BufferStoreStmt, IfStmt
from hidet.ir.functors import rewrite
from hidet.ir.layout import TaskLayout, DataLayout
from hidet.ir.node import Node
from hidet.ir.primitives import block_idx, thread_idx, syncthreads, printf
from hidet.ir.task import Task, Grid
from hidet.ir.type import TensorType
from hidet.utils import prod, cuda
from hidet.implement.common import VirtualTensor


class Conv2dSchedule(Schedule):
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
        self.regs_a_ldg_layout = local([block_threads // block_k, block_k]) * row_major([block_shape[0] // (block_threads // block_k), 1])
        self.regs_b_ldg_layout = row_major([1, block_shape[1] // (block_threads // block_k)]) * local([block_k, block_threads // block_k])

        # derived constants
        self.block_threads = block_threads
        self.block_shape = block_shape
        self.check((self.smem_a_layout.size + self.smem_b_layout.size) * 4 <= cuda.max_smem_bytes_per_block(), 'smem exceeded')

    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [('bwx', self.block_warps[0]),
                ('bwy', self.block_warps[1]),
                ('wox', self.warp_outer[0]),
                ('woy', self.warp_outer[1]),
                ('wmx', self.warp_middle[0]),
                ('wmy', self.warp_middle[1]),
                ('wix', self.warp_inner[0]),
                ('wiy', self.warp_inner[1]),
                ('bk', self.block_k)]

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return []

    def check(self, cond, msg=None):
        if not cond:
            raise NotSupportedError(msg)

    @staticmethod
    def schedules() -> List['Conv2dSchedule']:
        ret = []
        for block_warps in [[1, 1], [1, 2], [2, 1], [2, 2], [2, 4], [4, 2], [4, 4]]:
            for warp_outer in [[1, 1], [1, 2], [2, 1], [2, 2]]:
                for warp_middle in [[8, 4], [4, 8]]:
                    for warp_inner in [[4, 4]]:
                        for block_k in [4, 8, 16, 24]:
                            with contextlib.suppress(NotSupportedError):
                                ret.append(Conv2dSchedule(block_warps, warp_outer, warp_middle, warp_inner, block_k))
        return ret


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


def double(layout):
    return DataLayout.row_major([2]) + layout


@register_impl('cuda_grid_static_conv2d_implicit_gemm_implementer')
class CudaGridStaticConv2dImplicitGemmImplementer(Implementer):
    def __init__(self):
        self.pattern = Pattern()

    def priority(self) -> int:
        return 2

    def task_pattern(self) -> TaskPattern:
        return self.pattern.task_pattern

    def implement(self, task: Task, match: Mapping[Node, Any]) -> IRModule:
        search_schedule = False
        if search_schedule:
            schedules = Conv2dSchedule.schedules()
            ir_modules = [self.implement_schedule(task, match, schedule) for schedule in schedules]
            d: Pattern = pattern2matched(self.pattern, match)
            n, c, p, q, rc, rx, ry = d.n, d.c, d.p, d.q, d.rc, d.rx, d.ry
            task_label = 'conv2d_n_{}_c_{}_p_{}_q_{}_rc_{}_rx_{}_ry_{}'.format(n, c, p, q, rc, rx, ry)
            return self.resolve(task, match, schedules, ir_modules, task_label=task_label, parallel=True, verbose=True)
        else:
            default_schedule = Conv2dSchedule()
            return self.implement_schedule(task, match, default_schedule)

    def implement_schedule(self, task: Task, match: Mapping[Node, Any], schedule: Conv2dSchedule) -> IRModule:
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
            x = VirtualTensor(lambda i, k: rewrite(x_expr, {d.an: i // (p * q), d.ap: (i // q) % p, d.aq: i % q, d.arc: k // (rx * ry), d.arx: (k // ry) % rx, d.ary: k % ry}))
            w = VirtualTensor(lambda k, j: rewrite(w_expr, {d.ac: j, d.arc: k // (rx * ry), d.arx: (k // ry) % rx, d.ary: k % ry}))

            # local vars
            smem_a = Var(hint='smem_a', type=TensorType(scope='shared', dtype='float32', layout=double(schedule.smem_a_layout)))
            smem_b = Var(hint='smem_b', type=TensorType(scope='shared', dtype='float32', layout=double(schedule.smem_b_layout)))
            regs_a = Var(hint='regs_a', type=TensorType(scope='register', dtype='float32', layout=double(schedule.regs_a_layout)))
            regs_b = Var(hint='regs_b', type=TensorType(scope='register', dtype='float32', layout=double(schedule.regs_b_layout)))
            regs_c = Var(hint='regs_c', type=TensorType(scope='register', dtype='float32', layout=schedule.regs_c_layout))
            regs_a_ldg = Var(hint='regs_a_ldg', type=TensorType(scope='register', dtype='float32', layout=schedule.regs_a_ldg_layout))
            regs_b_ldg = Var(hint='regs_b_ldg', type=TensorType(scope='register', dtype='float32', layout=schedule.regs_b_ldg_layout))
            fb.extend_local_vars([smem_a, smem_b, regs_a, regs_b, regs_c, regs_a_ldg, regs_b_ldg])

            # body
            block_k_tiles = (gemm_k + block_k - 1) // block_k
            sb = StmtBuilder()
            for ii, jj in schedule.c_r2g_layout(thread_idx()):
                sb += BufferStoreStmt(regs_c, [ii, jj], 0.0)
            with sb.lets(['bi', 'bj'], grid_layout(block_idx())[0]) as (bi, bj):
                block_offset = [idx * dim for idx, dim in zip([bi, bj], schedule.block_shape)]
                # transfer first tile
                sb += self.copy(x[block_offset[0]:, :], regs_a_ldg, schedule.a_g2s_layout)
                sb += self.copy(regs_a_ldg, smem_a[0], schedule.a_g2s_layout)
                sb += self.copy(w[:, block_offset[1]:], regs_b_ldg, schedule.b_g2s_layout)
                sb += self.copy(regs_b_ldg, smem_b[0], schedule.b_g2s_layout)
                sb += syncthreads()
                sb += self.copy(smem_a[0], regs_a[0], schedule.a_s2r_layout)
                sb += self.copy(smem_b[0], regs_b[0], schedule.b_s2r_layout)
                sb += syncthreads()
                with sb.for_loop('bk', block_k_tiles-1) as k0:
                    block_offset_k = (k0 + 1) * block_k
                    with sb.for_loop('wk', schedule.block_k) as k1:
                        with sb.if_then(Equal(k1, schedule.block_k - 1)):
                            sb += self.copy(regs_a_ldg, smem_a[(k0 + 1) % 2], schedule.a_g2s_layout)
                            sb += self.copy(regs_b_ldg, smem_b[(k0 + 1) % 2], schedule.b_g2s_layout)
                            sb += syncthreads()
                            sb += self.copy(smem_a[(k0 + 1) % 2], regs_a[(k1 + 1) % 2], schedule.a_s2r_layout)
                            sb += self.copy(smem_b[(k0 + 1) % 2], regs_b[(k1 + 1) % 2], schedule.b_s2r_layout)
                        with sb.otherwise():
                            sb += self.copy(smem_a[k0 % 2, :, k1 + 1:], regs_a[(k1 + 1) % 2], schedule.a_s2r_layout)
                            sb += self.copy(smem_b[k0 % 2, k1 + 1:, :], regs_b[(k1 + 1) % 2], schedule.b_s2r_layout)
                        with sb.if_then(Equal(k1, 0)):
                            sb += self.copy(x[block_offset[0]:, block_offset_k:], regs_a_ldg, schedule.a_g2s_layout)
                            sb += self.copy(w[block_offset_k:, block_offset[1]:], regs_b_ldg, schedule.b_g2s_layout)
                        for ii, jj in schedule.c_mma_layout(thread_idx()):
                            sb += BufferStoreStmt(regs_c, [ii, jj], regs_c[ii, jj] + regs_a[k1 % 2, ii, 0] * regs_b[k1 % 2, 0, jj])
                with sb.let('k0', block_k_tiles - 1) as k0:
                    with sb.for_loop('k1', schedule.block_k) as k1:
                        with sb.if_then(k1 < schedule.block_k - 1):
                            sb += self.copy(smem_a[k0 % 2, :, k1 + 1:], regs_a[(k1 + 1) % 2], schedule.a_s2r_layout)
                            sb += self.copy(smem_b[k0 % 2, k1 + 1:, :], regs_b[(k1 + 1) % 2], schedule.b_s2r_layout)
                        for ii, jj in schedule.c_mma_layout(thread_idx()):
                            sb += BufferStoreStmt(regs_c, [ii, jj], regs_c[ii, jj] + regs_a[k1 % 2, ii, 0] * regs_b[k1 % 2, 0, jj])
                # c write back
                out = param_map[task.compute]
                for ii, jj in schedule.c_r2g_layout(thread_idx()):
                    gi = ii + bi * block_shape[0]
                    gj = jj + bj * block_shape[1]
                    with sb.if_then(And.join(gi < gemm_m, gj < gemm_n)):
                        sb += BufferStoreStmt(out, [gi // (p * q), gj, (gi // q) % p, gi % q], regs_c[ii, jj])
            fb.set_body(sb.finish())
        func = fb.get()
        return IRModule(funcs={func.name: func}, task=task)

    def copy(self, src, dst, layout, src_predicate=None, dst_predicate=None):
        sb = StmtBuilder()
        for indices in layout(thread_idx()):
            value = src.__getitem__(indices)
            if src_predicate:
                value = if_then_else(src_predicate(*indices), value, 0.0)
            stmt = BufferStoreStmt(dst, indices, value)
            if dst_predicate:
                stmt = IfStmt(dst_predicate(*indices), stmt)
            sb += stmt
        return sb.finish()
