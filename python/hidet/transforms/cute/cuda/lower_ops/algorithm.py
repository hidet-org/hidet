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
from typing import List, Union

from hidet.utils.py import gcd

from hidet.ir.tools import infer_type
from hidet.ir.expr import Expr, var
from hidet.ir.type import DataType, TensorType

from hidet.ir.cute.ops.algorithm import InclusiveScan
from hidet.ir.cute import (
    size,
    TensorLayout,
    composition,
    left_inverse,
    make_layout,
    flatten,
    coalesce,
    compact_col_major,
    common_reshape,
    idx2crd,
)
from hidet.ir.cute.type import TiledTensorType, TiledTensorLayout

from ..instruction_selection import memory_instructions
from .registry import OpEmitter, Buffer, register_impl


@register_impl(InclusiveScan)
class InclusiveScanEmitter(OpEmitter):
    op2exec_plan = {}

    def try_schedule_thread_scan(self, shape: List[int], thr_layout: TensorLayout, val_layout: TensorLayout, axis: int):
        num_threads = thr_layout.size()
        shape_to_thrval = left_inverse(make_layout(thr_layout, val_layout))
        shape_to_thrval = composition(shape_to_thrval, TensorLayout(tuple(shape)))
        scan_shape_to_thrval = shape_to_thrval[axis]
        scan_stride = flatten(scan_shape_to_thrval.stride_tuple)
        if any(d < num_threads for d in scan_stride):
            return None
        cont_stride = compact_col_major(tuple(shape))
        flat_val_shape = flatten(val_layout.shape_tuple)
        flat_val_stride = flatten(val_layout.stride_tuple)
        result_shape = []
        par_val_stride = []
        scan_val_stride = []
        lo = cont_stride[axis]
        hi = lo * shape[axis]
        for s, d in zip(flat_val_shape, flat_val_stride):
            if lo < d <= hi:
                if s * d <= hi:
                    result_shape.append(s)
                    par_val_stride.append(0)
                    scan_val_stride.append(d)
                else:
                    s1 = hi // d
                    s2 = s // s1
                    result_shape.append(s1)
                    par_val_stride.append(0)
                    scan_val_stride.append(d)
                    result_shape.append(s2)
                    par_val_stride.append(s1 * d)
                    scan_val_stride.append(0)
            elif d <= lo:
                if s * d <= lo:
                    result_shape.append(s)
                    par_val_stride.append(d)
                    scan_val_stride.append(0)
                elif s * d <= hi:
                    s1 = lo // d
                    result_shape.append(s1)
                    par_val_stride.append(d)
                    scan_val_stride.append(0)
                    s2 = s // s1
                    result_shape.append(s2)
                    par_val_stride.append(0)
                    scan_val_stride.append(lo)
                else:
                    s1 = lo // d
                    result_shape.append(s1)
                    par_val_stride.append(d)
                    scan_val_stride.append(0)
                    s2 = hi // lo
                    result_shape.append(s2)
                    par_val_stride.append(0)
                    scan_val_stride.append(lo)
                    s3 = s * d // hi
                    result_shape.append(s3)
                    par_val_stride.append(d)
                    scan_val_stride.append(hi)
            else:
                result_shape.append(s)
                par_val_stride.append(d)
                scan_val_stride.append(0)
        cont_stride = compact_col_major(tuple(result_shape))
        par_stride, par_shape, _ = list(
            zip(*list(filter(lambda x: x[2] != 0, zip(cont_stride, result_shape, par_val_stride))))
        )
        scan_coord_stride, scan_shape, scan_stride = list(
            zip(*list(sorted(filter(lambda x: x[0] != 0, zip(scan_val_stride, result_shape, cont_stride)))))
        )
        par_layout = coalesce(TensorLayout(par_shape, par_stride))
        scan_layout = coalesce(TensorLayout(scan_shape, scan_stride))
        scan_coord_layout = coalesce(TensorLayout(scan_shape, scan_coord_stride))
        scan_layout, scan_coord_layout = common_reshape(scan_layout, scan_coord_layout)
        return par_layout, scan_layout, scan_coord_layout

    def schedule(
        self,
        dtype: DataType,
        shape: List[int],
        thr_layout: TensorLayout,
        val_layout: TensorLayout,
        axis: int,
        max_shared_memory_size: int = 65536 * 2,
    ):
        num_threads = thr_layout.size()
        num_scan_length = shape[axis]
        assert (num_scan_length & (num_scan_length - 1)) == 0 and num_scan_length > 0
        assert num_scan_length <= num_threads and num_threads % num_scan_length == 0
        sche = self.try_schedule_thread_scan(shape, thr_layout, val_layout, axis)
        if sche is not None:
            return "thread_scan", tuple(sche)

        max_processing_elements = max_shared_memory_size // dtype.nbytes
        total_size = size(tuple(shape))

        assert total_size % num_threads == 0
        max_working_batches = gcd(max_processing_elements // num_scan_length, total_size // num_scan_length)
        inner_shape = []
        inner_stride = []
        outer_shape = []
        outer_stride = []
        current_index = 1
        for i, s in enumerate(shape):
            if i != axis:
                g = gcd(s, max_working_batches)
                max_working_batches //= g
                inner_shape.append(g)
                outer_shape.append(s // g)
            else:
                inner_shape.append(num_scan_length)
                outer_shape.append(s // num_scan_length)
            inner_stride.append(current_index)
            outer_stride.append(current_index * inner_shape[-1])
            current_index *= s
        assert max_working_batches == 1

        shared_memory_shape = []
        shared_memory_stride = []
        current_index = num_scan_length
        for i, (s, inner_s) in enumerate(zip(shape, inner_shape)):
            shared_memory_shape.append(s)
            if i == axis:
                shared_memory_stride.append(1)
            else:
                shared_memory_stride.append(current_index)
                current_index *= inner_s

        shape_to_thrval = left_inverse(make_layout(thr_layout, val_layout))
        inner_to_thrval = coalesce(composition(shape_to_thrval, TensorLayout(tuple(inner_shape), tuple(inner_stride))))
        outer_to_thrval = coalesce(composition(shape_to_thrval, TensorLayout(tuple(outer_shape), tuple(outer_stride))))

        num_values = val_layout.size()
        inner_thread_shape = []
        inner_thread_stride = []
        outer_thread_shape = []
        outer_thread_stride = []
        inner_value_shape = []
        inner_value_stride = []
        outer_value_shape = []
        outer_value_stride = []
        current_thread_index = 1
        current_value_index = 1
        sorted_sd = sorted(
            zip(flatten(inner_to_thrval.shape_tuple), flatten(inner_to_thrval.stride_tuple)), key=lambda x: x[1]
        )
        for s, d in sorted_sd:
            if d < num_threads:
                if d > current_thread_index:
                    assert d % current_thread_index == 0
                    inner_thread_shape.append(d // current_thread_index)
                    inner_thread_stride.append(0)
                    outer_thread_shape.append(d // current_thread_index)
                    outer_thread_stride.append(current_thread_index)
                    current_thread_index = d
                if s * d < num_threads:
                    inner_thread_shape.append(s)
                    outer_thread_shape.append(s)
                else:
                    inner_thread_shape.append(num_threads // d)
                    outer_thread_shape.append(num_threads // d)
                    s1 = s // (num_threads // d)
                    inner_value_shape.append(s1)
                    inner_value_stride.append(current_value_index)
                    outer_value_shape.append(s1)
                    outer_value_stride.append(0)
                    current_value_index *= s1
                inner_thread_stride.append(current_thread_index)
                outer_thread_stride.append(0)
                current_thread_index *= inner_thread_shape[-1]
            else:
                val = d // num_threads
                if val > current_value_index:
                    assert val % current_value_index == 0
                    outer_value_shape.append(val // current_value_index)
                    outer_value_stride.append(current_value_index)
                    inner_value_shape.append(val // current_value_index)
                    inner_value_stride.append(0)
                    current_value_index = val
                inner_value_shape.append(s)
                outer_value_shape.append(s)
                inner_value_stride.append(current_value_index)
                outer_value_stride.append(0)
                current_value_index *= s
        if current_thread_index < num_threads:
            outer_thread_shape.append(num_threads // current_thread_index)
            outer_thread_stride.append(current_thread_index)
            inner_thread_shape.append(num_threads // current_thread_index)
            inner_thread_stride.append(0)
        if current_value_index < num_values:
            outer_value_shape.append(num_values // current_value_index)
            outer_value_stride.append(current_value_index)
            inner_value_shape.append(num_values // current_value_index)
            inner_value_stride.append(0)
        inner_thread_layout = coalesce(TensorLayout(tuple(inner_thread_shape), tuple(inner_thread_stride)))
        outer_thread_layout = coalesce(TensorLayout(tuple(outer_thread_shape), tuple(outer_thread_stride)))
        inner_value_layout = coalesce(TensorLayout(tuple(inner_value_shape), tuple(inner_value_stride)))
        outer_value_layout = coalesce(TensorLayout(tuple(outer_value_shape), tuple(outer_value_stride)))

        shared_memory_layout = TensorLayout(tuple(shared_memory_shape), tuple(shared_memory_stride))
        inner_thr_layout = coalesce(composition(thr_layout, inner_thread_layout))
        inner_val_layout = coalesce(composition(val_layout, inner_value_layout))

        shared_memory_size = size(tuple(inner_shape)) * dtype.nbytes

        return "block_scan", (
            outer_to_thrval,
            outer_thread_layout,
            inner_value_layout,
            outer_value_layout,
            inner_thr_layout,
            inner_val_layout,
            shared_memory_layout,
            shared_memory_size,
        )

    def _instruction_selection(self, src: Buffer, dst: Buffer):
        for inst in memory_instructions:
            result = inst.match(src, dst)
            if result is not None:
                return inst, *result
        raise NotImplementedError(f"no instruction found for {src} and {dst}")

    def request_smem_nbytes(self, op: InclusiveScan) -> int:
        x_type = infer_type(op.x)
        if not isinstance(x_type, TiledTensorType):
            raise TypeError(f"inclusive_scan op should be applied on TiledTensorType.(got:{x_type})")
        if not isinstance(x_type.layout, TiledTensorLayout):
            raise TypeError(f"inclusive_scan op should be applied on TiledTensorLayout.(got:{x_type.layout})")
        dtype = x_type.dtype
        shape = x_type.layout.shape()
        thr_layout = x_type.layout.thr_layout()
        val_layout = x_type.layout.val_layout()
        axis = op.axis
        if op in self.op2exec_plan:
            algo, plan = self.op2exec_plan[op]
        else:
            algo, plan = self.schedule(dtype, shape, thr_layout, val_layout, axis)
            self.op2exec_plan[op] = (algo, plan)
        if algo == "thread_scan":
            return 0
        else:
            _, _, _, _, _, _, _, shared_memory_size = plan
            return shared_memory_size

    def emit(self, op: InclusiveScan, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        assert isinstance(args[1], (Expr, Buffer))
        assert isinstance(output, Buffer)
        x_arg = args[0]
        dtype = x_arg.dtype
        x_layout = x_arg.layout
        assert isinstance(x_layout, TiledTensorLayout)
        shape = x_layout.shape()
        thr_layout = x_layout.thr_layout()
        val_layout = x_layout.val_layout()
        axis = op.axis
        init_arg = args[1]
        if isinstance(init_arg, Buffer):
            init_layout = init_arg.layout
            assert isinstance(init_layout, TiledTensorLayout)

        src_buf = x_arg.buffer
        dst_buf = output.buffer
        src_ty = infer_type(src_buf)
        if isinstance(src_ty, TensorType):
            src_buf = ~src_buf[0]
        dst_buf = ~dst_buf[0]

        # if "group_ids" in op.annotations:
        #     group_ids = op.annotations["group_ids"]
        #     assert "group_threads" in op.annotations
        #     group_threads = op.annotations["group_threads"]
        #     tid = tid_in_groups(group_ids)
        #     sync = bar_sync(group_threads)
        # else:
        #     tid = threadIdx.x
        #     sync = syncthreads()
        # num_threads = thr_layout.size()
        if op in self.op2exec_plan:
            algo, plan = self.op2exec_plan[op]
        else:
            algo, plan = self.schedule(dtype, shape, thr_layout, val_layout, axis)
            self.op2exec_plan[op] = (algo, plan)

        if algo == "thread_scan":
            par_layout, scan_layout, scan_coord_layout = plan

            num_iters_par = par_layout.size()
            num_iters_scan = scan_layout.size()

            with self.for_grid(num_iters_par) as j:
                if isinstance(init_arg, Buffer):
                    running_prefix = init_arg.buffer[j]
                else:
                    raise NotImplementedError(f"init_arg is not a Buffer: {init_arg}")
                running_var_ty = infer_type(running_prefix)
                running_var = var("running_prefix", running_var_ty)
                self.declare(running_var, running_prefix)
                with self.for_grid(num_iters_scan) as i:
                    index = j + scan_layout(i)
                    if op.scan_length is not None:
                        coords = idx2crd(scan_coord_layout(i), shape)
                        with self.if_then(coords[axis] < op.scan_length):
                            self.assign(running_var, op.scan_op(running_var, x_arg.buffer[index]))
                            self.buffer_store(output.buffer, [index], running_var)
                    else:
                        self.assign(running_var, op.scan_op(running_var, x_arg.buffer[index]))
                        self.buffer_store(output.buffer, [index], running_var)
                if op.update_init:
                    self.buffer_store(init_arg.buffer, [j], running_var)
            return
        raise NotImplementedError(f"algo: {algo}")

        # XiaoZhang: The code below is commented out because in the selective_scan operator,
        # the block_scan is not as efficient as the thread_scan. As a result, we currently
        # use the thread_scan for the selective_scan operator. I want to keep the code
        # of block_scan for future use.

        # (
        #     outer_to_thrval,
        #     outer_thread_layout,
        #     inner_value_layout,
        #     outer_value_layout,
        #     inner_thr_layout,
        #     inner_val_layout,
        #     shared_memory_layout,
        #     shared_memory_size,
        # ) = plan

        # thread_memory_layout = coalesce(composition(shared_memory_layout, inner_thr_layout))
        # inner_memory_layout = coalesce(composition(shared_memory_layout, inner_val_layout))

        # smem_addr = var("smem", ~dtype)
        # self.declare(smem_addr, self.get_smem_ptr(op, dtype, 4))
        # sts, sts_src_layout, sts_dst_layout = self._instruction_selection(
        #     Buffer(src_buf, None, dtype, inner_value_layout, "register"),
        #     Buffer(smem_addr, None, dtype, inner_memory_layout, "shared"),
        # )
        # lds, lds_src_layout, lds_dst_layout = self._instruction_selection(
        #     Buffer(smem_addr, None, dtype, inner_memory_layout, "shared"),
        #     Buffer(dst_buf, None, dtype, inner_value_layout, "register"),
        # )
        # num_outer_iters = outer_to_thrval.size()

        # with self.for_grid([num_outer_iters]) as j:
        #     thrval = outer_to_thrval(j)
        #     working_thread_index = thrval % num_threads
        #     current_thread_index = outer_thread_layout(tid)
        #     value_index = thrval // num_threads
        #     src_addr_base = var("src_addr_base", ~dtype)
        #     smem_addr_base = var("smem_addr_base", ~dtype)
        #     self.declare(src_addr_base, src_buf + outer_value_layout(value_index, base=x_arg.offset))
        #     self.declare(smem_addr_base, smem_addr + thread_memory_layout(tid))
        #     with self.if_then(working_thread_index == current_thread_index):
        #         with self.for_grid(sts_src_layout[1].shape) as sts_coords:
        #             src_ptr = var("src_ptr", ~dtype)
        #             smem_ptr = var("smem_ptr", ~dtype)
        #             self.declare(src_ptr, src_addr_base + sts_src_layout[1](sts_coords))
        #             self.declare(smem_ptr, smem_addr_base + sts_dst_layout[1](sts_coords))
        #             self.append(sts(src_ptr, smem_ptr))
        #     self.append(sync)

        #     dst_addr_base = var("dst_addr_base", ~dtype)
        #     self.declare(dst_addr_base, dst_buf + outer_value_layout(value_index, base=output.offset))
        #     self.assign(smem_addr_base, smem_addr + thread_memory_layout(tid))
        #     with self.if_then(working_thread_index == current_thread_index):
        #         with self.for_grid(lds_src_layout[1].shape) as lds_coords:
        #             smem_ptr = var("smem_ptr", ~dtype)
        #             dst_ptr = var("dst_ptr", ~dtype)
        #             self.declare(smem_ptr, smem_addr_base + lds_src_layout[1](lds_coords))
        #             self.declare(dst_ptr, dst_addr_base + lds_dst_layout[1](lds_coords))
        #             self.append(lds(smem_ptr, dst_ptr))
        #     with self.if_then(j < num_outer_iters - 1):
        #         self.append(sync)
