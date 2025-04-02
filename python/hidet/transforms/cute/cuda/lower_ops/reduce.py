# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Union
from hidet.ir.expr import Expr, tensor_var, cast, deref, logical_and, logical_or
from hidet.ir.type import PointerType, TensorType
from hidet.ir.tools import infer_type
from hidet.lang.cuda import threadIdx
from hidet.ir.primitives.cuda.sync import bar_sync
from hidet.ir.primitives.cuda import shfl_xor_sync, syncthreads
from hidet.ir.dtypes import f64, u32, f16
from hidet.ir.cute.ops.reduce import Reduce
from hidet.ir.cute.type import TiledTensorType
from hidet.ir.cute import (
    TiledTensorLayout,
    TensorLayout,
    coalesce,
    filter,
    flatten,
    repeat_like,
    compact_col_major,
    group,
    make_layout,
    idx2crd,
)
from hidet.ir.cute.contexts import tid_in_groups
from .registry import OpEmitter, Buffer, register_impl


WARP_SIZE = 32


@register_impl(Reduce)
class ReduceEmitter(OpEmitter):
    """
    Emitter for Reduce operations, responsible for generating code for reduction operations on tensors.
    """

    def _require_inter_warp_reduce(self, warp: TensorLayout):
        """
        Determine if inter-warp reduction is required based on the warp layout.

        Args:
            warp (TensorLayout): The layout of the warp.

        Returns:
            bool: True if inter-warp reduction is required, False otherwise.
        """

        return 0 in coalesce(warp).stride_tuple

    def request_smem_nbytes(self, op: Reduce) -> int:
        """
        Request the number of shared memory bytes required for the reduction operation.

        Args:
            op (Reduce): The reduction operation.

        Returns:
            int: The number of shared memory bytes required.
        """

        x_type = infer_type(op.x)
        if not isinstance(x_type, TiledTensorType):
            raise TypeError(f"reduce op should be applied on TiledTensorType.(got:{x_type})")
        if not isinstance(x_type.layout, TiledTensorLayout):
            raise TypeError(f"reduce op should be applied on TiledTensorLayout.(got:{x_type.layout})")
        dst_type = op.infer_type([x_type])
        dst_thr = dst_type.layout.thr_layout()
        dst_val = dst_type.layout.val_layout()
        lane, warp = group(dst_thr, WARP_SIZE)
        if self._require_inter_warp_reduce(warp):
            return filter(lane).size() * warp.size() * dst_val.count() * x_type.dtype.nbits // 8
        else:
            return 0

    def canonicalize(self, val: TensorLayout):
        """
        Canonicalize the value layout. Typically, the value layout represents the layout of a register
        tensor held by a threads. The canonicalization process skips the 0-strides and generates
        a contiguous strides for the tensor, so that we could use this layout in the code generation to
        access elements in this tensor.

        Args:
            val (TensorLayout): The tensor layout.

        Returns:
            Tuple[List[int], List[int]]: The canonicalized shape and stride.
        """

        shape = list(flatten(val.shape_tuple))
        stride = list(flatten(val.stride_tuple))
        current = 1
        for i, (s, d) in enumerate(zip(shape, stride)):
            if d != 0:
                stride[i] = current
                current = current * s
        return shape, stride

    def canonicalize_val(self, src_val: TensorLayout, dst_val: TensorLayout):
        """
        Canonicalize the value layouts of the source and destination tensors. This process seperates the axes
        into two groups: parallel axes and reduction axes. The parallel axes are the axes that both source and
        destination tensors have the non-zero strides, while the reduction axes are the axes that the destination
        tensor has the zero strides. After re-grouping the axes, we can easily generate code for the intra-thread
        reduction.

        For example,
        - src_val = (2, 2, 2, 2):(1, 2, 4, 8)
        - dst_val = (2, 2, 2, 2):(1, 0, 2, 0)
        parallel axes: (2, 2):(1, 4) and (2, 2):(1, 2)
        reduction axes: (2, 2):(2, 8) and (2, 2):(0, 0)
        generated code:
        ```python
        for par in grid(par.shape):
            dst[par, 0] = init()
            for red in grid(red.shape):
                dst[par, 0] = reduce(dst[par, 0], src[par, red])
        ```

        Args:
            src_val (TensorLayout): The source tensor layout.
            dst_val (TensorLayout): The destination tensor layout.

        Returns:
            Tuple[Tuple[int], Tuple[int], TensorLayout, TensorLayout]: The canonicalized shapes and layouts
            for source and destination.
        """

        src_shape, src_stride = self.canonicalize(src_val)
        dst_shape, dst_stride = self.canonicalize(dst_val)
        par_shape = []
        par_stride_src = []
        par_stride_dst = []
        red_shape = []
        red_stride_src = []
        red_stride_dst = []
        for ss, sd, ds, dd in zip(src_shape, src_stride, dst_shape, dst_stride):
            if dd == 0 and sd != 0:
                if ss != ds:
                    raise TypeError(f"invalid value layout for reduce op(src:{src_val},dst:{dst_val})")
                red_shape.append(ss)
                red_stride_src.append(sd)
                red_stride_dst.append(dd)
            else:
                if ss != ds:
                    raise TypeError(f"invalid value layout for reduce op(src:{src_val},dst:{dst_val})")
                par_shape.append(ss)
                par_stride_src.append(sd)
                par_stride_dst.append(dd)
        src_par = TensorLayout(tuple(par_shape), tuple(par_stride_src))
        src_red = TensorLayout(tuple(red_shape), tuple(red_stride_src))
        src_layout = make_layout(src_par, src_red)
        dst_par = TensorLayout(tuple(par_shape), tuple(par_stride_dst))
        dst_red = TensorLayout(tuple(red_shape), tuple(red_stride_dst))
        dst_layout = make_layout(dst_par, dst_red)
        return tuple(par_shape), tuple(red_shape), src_layout, dst_layout

    def _get_lds_sts(self, nr_bits: int):
        """
        Get the appropriate load/store instructions based on the number of bits.

        Args:
            nr_bits (int): The number of bits.

        Returns:
            Tuple[Callable, Callable, int]: The load/store instructions and bits per instruction.
        """

        from hidet.ir.primitives import lds128, lds64, lds32, lds16, lds8, sts128, sts64, sts32, sts16, sts8

        if nr_bits % 128 == 0:
            lds = lds128
            sts = sts128
            bits_per_inst = 128
        elif nr_bits % 64 == 0:
            lds = lds64
            sts = sts64
            bits_per_inst = 64
        elif nr_bits % 32 == 0:
            lds = lds32
            sts = sts32
            bits_per_inst = 32
        elif nr_bits % 16 == 0:
            lds = lds16
            sts = sts16
            bits_per_inst = 16
        elif nr_bits % 8 == 0:
            lds = lds8
            sts = sts8
            bits_per_inst = 8
        else:
            lds = None
            sts = None
            bits_per_inst = None
        return lds, sts, bits_per_inst

    def emit(self, op: Reduce, args: List[Union[Buffer, Expr]], output: Buffer):
        """
        Emit the code for the reduction operation.

        Args:
            op (Reduce): The reduction operation.
            args (List[Union[Buffer, Expr]]): The input arguments.
            output (Buffer): The output buffer.
        """

        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        assert isinstance(src.layout, TiledTensorLayout)
        # intra - thread reduce
        src_val = src.layout.val_layout()
        dst_val = dst.layout.val_layout()
        from hidet.ir.cute.layout import common_reshape

        src_val, dst_val = common_reshape(src_val, dst_val)
        src_buf = src.buffer
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, (TensorType, PointerType))
        if isinstance(src_ty, TensorType):
            #    src_buf = ~src.buffer[0]
            src_dtype = src_ty.dtype
        else:
            src_dtype = src_ty.base_type

        # reduce_per_thread = self.auto_var(
        #    v=tensor_var(hint="reduce_per_thread", shape=[dst_val.count()], dtype=src_dtype)
        # )
        # re-group the axes into parallel and reduction axes and then generate code for the intra-thread reduction.
        # for details, see the docstring of the `canonicalize_val` method.
        par, red, src_layout, dst_layout = self.canonicalize_val(src_val, dst_val)
        zeros = repeat_like(red)
        with self.for_grid(par) as outer:
            self.buffer_store(dst.buffer, [dst_layout((outer, zeros))], op.init())
            with self.for_grid(red) as inner:
                self.buffer_store(
                    dst.buffer,
                    [dst_layout((outer, zeros))],
                    op.op(
                        dst.buffer[dst_layout((outer, zeros), base=dst.offset)],
                        src_buf[src_layout((outer, inner), base=src.offset)],
                    ),
                )

        # intra - warp reduce

        # src_thr = src.layout.thr_layout()
        # generates the warp shuffle instructions to perform the intra-warp reduction.
        dst_thr = dst.layout.thr_layout()
        lane, warp = group(dst_thr, WARP_SIZE)
        flat_shape = flatten(lane.shape_tuple)
        flat_stride = flatten(lane.stride_tuple)
        costride = compact_col_major(flat_shape)
        deltas = []
        for s, d, d1 in zip(reversed(flat_shape), reversed(flat_stride), reversed(costride)):
            # 0-stride axis indicate we need to reduce over this axis
            # the co-stride indicates the lower-bound of the loop
            # for example: if the thread layout is (4, 8):(0, 4), the co-stride is (1, 4)
            # we need to reduce the axis with 0-stride and the co-stride is 4
            # so the upper-bound of the loop is 1 * 4 // 2 = 2
            # the lower-bound of the loop is 1
            # the deltas are 2, 1
            # the generated code is:
            # for delta in [2, 1]:
            #     temp = shfl_xor_sync(0xFFFFFFFF, temp, delta)
            #     dst[...] = op(dst[...], temp)
            # dst is the local register tensor held by each thread
            if d == 0:
                end = d1 * s // 2
                start = d1
                while end >= start:
                    deltas.append(end)
                    end //= 2
        nr_regs = dst_val.count()
        nr_bits = nr_regs * src_dtype.nbits
        if nr_bits % 64 == 0:
            shfl_dtype = f64
        elif nr_bits % 32 == 0:
            shfl_dtype = u32
        elif nr_bits % 16 == 0:
            shfl_dtype = f16
        else:
            raise NotImplementedError(f"unable to shuffle data within a warp.(reduce_per_thread:{dst_val})")
        temp = self.auto_var(v=tensor_var(hint="temp", shape=[nr_regs], dtype=src_dtype))
        for delta in deltas:
            iters = nr_bits // shfl_dtype.nbits
            elems_per_iter = shfl_dtype.nbits // src_dtype.nbits
            with self.for_grid(iters) as i:
                val = self.auto_var(hint="val", e=deref(cast(~dst.buffer[i * elems_per_iter], ~shfl_dtype)))
                tmp = self.auto_var(hint="tmp", e=cast(~temp[i * elems_per_iter], ~shfl_dtype))
                self.buffer_store(tmp, [0], shfl_xor_sync(u32(0xFFFFFFFF), val, delta))
            with self.for_grid(nr_regs) as i:
                self.buffer_store(dst.buffer, [i], op.op(dst.buffer[i], temp[i]))

        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            assert "group_threads" in op.annotations
            group_threads = op.annotations["group_threads"]
            tid = tid_in_groups(group_ids)
            sync = bar_sync(group_threads)
        else:
            tid = threadIdx.x
            sync = syncthreads()

        # inter - warp reduce
        # finally, we perform the inter-warp reduction if necessary.
        # the approach is straightforward:
        # re-group the warp layout into parallel and reduction axes
        # warps belonging to the parallel axes first store local data into shared memory.
        # then warps belonging to the reduction axes load the data from the shared memory to the local register tensor
        # and perform the reduction.
        if not self._require_inter_warp_reduce(warp):
            return
        self.append(sync)

        smem_size = filter(lane).size() * warp.size() * nr_regs
        smem_addr = self.auto_var(
            hint="smem_addr", e=self.get_smem_ptr(op, src_dtype, smem_size * src_dtype.nbits // 8)
        )
        lane_shape, lane_stride = self.canonicalize(lane)
        thread_shape = lane_shape + [warp.size()]
        thread_stride = lane_stride + [filter(lane).size()]
        thread_stride = [d * nr_regs for d in thread_stride]
        thread_layout = TensorLayout(tuple(thread_shape), tuple(thread_stride))
        lane_id = self.auto_var(hint="lane_id", e=tid % WARP_SIZE)
        lds, sts, bits_per_inst = self._get_lds_sts(nr_bits)
        if lds is None:
            raise NotImplementedError(f"cannot find lds/sts instruction to perform inter-warp reduce.(op:{op}")
        iters = nr_bits // bits_per_inst
        elems_per_iter = bits_per_inst // src_dtype.nbits
        BITS_PER_GPR = 32
        incr = BITS_PER_GPR // src_dtype.nbits
        smem_addr1 = self.auto_var(hint="smem_addr1", e=smem_addr + thread_layout(tid))
        lane_crds = idx2crd(lane_id, tuple(lane_shape))
        lane_crds_ = []
        for crd, d in zip(lane_crds, lane_stride):
            if d == 0:
                lane_crds_.append(crd)
        lane_cond = [crd == 0 for crd in lane_crds_]
        with self.if_then(logical_and(*lane_cond)):
            with self.for_grid(iters) as i:
                operands = [~dst.buffer[i * elems_per_iter + delta] for delta in range(0, elems_per_iter, incr)]
                operands.append(smem_addr1 + i * elems_per_iter)
                self.append(sts(*operands))

        self.append(sync)
        warp_id = self.auto_var(hint="warp_id", e=tid // WARP_SIZE)
        crds = idx2crd(warp_id, warp.shape_tuple)
        crds_ = []
        red_shape_warp = []
        red_stride_warp = []
        current = WARP_SIZE
        for crd, s, d in zip(crds, warp.shape_tuple, warp.stride_tuple):
            if d == 0:
                crds_.append(crd)
                red_shape_warp.append(s)
                red_stride_warp.append(current)
            current = current * s
        red_layout = TensorLayout(tuple(red_shape_warp), tuple(red_stride_warp))
        cond = [crd == 0 for crd in crds_] + lane_cond
        with self.if_then(logical_and(*cond)):
            with self.for_grid(red_shape_warp) as red:
                red_cond = [crd != 0 for crd in red] if isinstance(red, list) else [red != 0]
                with self.if_then(logical_or(*red_cond)):
                    smem_addr2 = self.auto_var(
                        hint="smem_addr2", e=smem_addr + thread_layout(lane_id + red_layout(red))
                    )
                    with self.for_grid(iters) as i:
                        operands = [~temp[i * elems_per_iter + delta] for delta in range(0, elems_per_iter, incr)]
                        operands += [smem_addr2 + i * elems_per_iter]
                        self.append(lds(*operands))
                    with self.for_grid(nr_regs) as i:
                        self.buffer_store(dst.buffer, [i], op.op(dst.buffer[i], temp[i]))
                    with self.for_grid(iters) as i:
                        operands = [~dst.buffer[i * elems_per_iter + delta] for delta in range(0, elems_per_iter, incr)]
                        operands.append(smem_addr1 + i * elems_per_iter)
                        self.append(sts(*operands))
        self.append(sync)
        thread_shape = lane_shape
        thread_stride = lane_stride
        current = filter(lane).size()
        for s, d in zip(warp.shape_tuple, warp.stride_tuple):
            if d == 0:
                thread_shape.append(s)
                thread_stride.append(d)
            else:
                thread_shape.append(s)
                thread_stride.append(current)
            current *= s
        thread_stride = [d * nr_regs for d in thread_stride]
        thread_layout = TensorLayout(tuple(thread_shape), tuple(thread_stride))
        smem_addr3 = self.auto_var(hint="smem_addr3", e=smem_addr + thread_layout(tid))
        with self.for_grid(iters) as i:
            operands = [~dst.buffer[i * elems_per_iter + delta] for delta in range(0, elems_per_iter, incr)]
            operands += [smem_addr3 + i * elems_per_iter]
            self.append(lds(*operands))
