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
from typing import Tuple, List, Union
from hidet.ir.expr import Expr, var
from hidet.ir.type import PointerType, TensorType
from hidet.ir.tools import infer_type
from hidet.lang.cuda import threadIdx, syncthreads

from hidet.ir.cute.ops.rearrange import Rearrange
from hidet.ir.cute.int_tuple import flatten, compact_col_major
from hidet.ir.cute.type import TiledTensorType
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout, composition, coalesce, make_layout, common_reshape

from ..instruction_selection import memory_instructions
from .registry import OpEmitter, Buffer, register_impl


@register_impl(Rearrange)
class RearrangeEmitter(OpEmitter):
    def __init__(self):
        super().__init__()
        self.op2exec_plan = {}

    def request_smem_nbytes(self, op: Rearrange) -> int:
        src = op.args[0]
        src_ty = infer_type(src)
        if op in self.op2exec_plan:
            (_, _, _, _, sts_layout, lds_layout) = self.op2exec_plan[op]
        else:
            assert isinstance(src_ty, TiledTensorType) and isinstance(src_ty.layout, TiledTensorLayout)
            shape = src_ty.layout.shape()
            src_thr_layout = src_ty.layout.thr_layout()
            src_val_layout = src_ty.layout.val_layout()
            _shape = op.layout.shape()
            dst_thr_layout = op.layout.thr_layout()
            dst_val_layout = op.layout.val_layout()
            if not shape == _shape:
                raise TypeError(f"Shape mismatch. (got:src({shape}),dst({_shape}))")
            src_tv_layout = make_layout(src_thr_layout, src_val_layout)
            dst_tv_layout = make_layout(dst_thr_layout, dst_val_layout)
            # print("HIT")
            # print(src_tv_layout, dst_tv_layout)
            self.op2exec_plan[op] = self._schedule(shape, src_tv_layout, dst_tv_layout)
            (_, _, _, _, sts_layout, lds_layout) = self.op2exec_plan[op]
        smem_size = sts_layout.cosize()
        assert (
            smem_size == lds_layout.cosize()
        ), f"Schedule failed due to shared memory size mismatch. (got:sts({smem_size}), lds({lds_layout.cosize()}))"
        return smem_size * src_ty.dtype.nbytes

    def _instruction_selection(self, src: Buffer, dst: Buffer):
        candidates = []
        for inst in memory_instructions:
            result = inst.match(src, dst)
            if result is not None:
                candidates.append((inst, *result))
        candidates = sorted(candidates, key=lambda x: -x[0].bytes_per_inst)
        return candidates[0]

    def _calc_inner_outer_layout(self, outer, inner, ref_stride):
        if isinstance(outer, list):
            outer = tuple(outer)
        if isinstance(inner, list):
            inner = tuple(inner)
        shape = []
        for o, i in zip(outer, inner):
            shape.append(i)
            shape.append(o)
        shape = tuple(shape)
        stride = compact_col_major(shape)
        ostrd = []
        istrd = []
        for idx in range(0, len(shape), 2):
            istrd.append(stride[idx])
            ostrd.append(stride[idx + 1])
        istrd = tuple(istrd)
        ostrd = tuple(ostrd)
        sorted_DS = sorted(zip(ref_stride, outer, ostrd))
        outer_shape = tuple(s for _, s, _ in sorted_DS)
        outer_stride = tuple(d for _, _, d in sorted_DS)
        return coalesce(TensorLayout(inner, istrd)), coalesce(TensorLayout(outer_shape, outer_stride))

    def _calc_smem_layout(self, shape, orig_layout, inner):
        if isinstance(inner, list):
            inner = tuple(inner)
        thr_layout = orig_layout[0]
        new_shp = thr_layout.shape_tuple + inner
        row_major = TensorLayout(shape, (shape[1], 1))
        new_layout = composition(row_major, TensorLayout(new_shp, flatten(orig_layout.stride)))
        orig_layout = composition(row_major, orig_layout)
        sorted_DS = sorted(zip(flatten(orig_layout.stride), flatten(orig_layout.shape), flatten(new_layout.shape)))
        shp = tuple(s for _, s, _ in sorted_DS)
        strd = compact_col_major(tuple(s for _, _, s in sorted_DS))
        # print("calc smem")
        # print(f"inner{inner}")
        # print(f"orig:{orig_layout}")
        return coalesce(composition(TensorLayout(shp, strd), row_major))

    def _schedule(self, shape, src: TensorLayout, dst: TensorLayout):
        src_thr_layout = src[0]
        src_val_layout = src[1]
        dst_thr_layout = dst[0]
        dst_val_layout = dst[1]
        st_shape = src_thr_layout.shape_tuple
        st_stride = src_thr_layout.stride_tuple
        dv_shape = dst_val_layout.shape_tuple
        dv_stride = dst_val_layout.stride_tuple
        sv_shape = src_val_layout.shape_tuple
        sv_stride = src_val_layout.stride_tuple
        dt_shape = dst_thr_layout.shape_tuple
        dt_stride = dst_thr_layout.stride_tuple

        dst_inner = []
        dst_outer = []
        src_inner = []
        src_outer = []
        data = []
        src_readers = {}
        dst_readers = {}
        for i, (s, d) in enumerate(zip(dv_shape, dv_stride)):
            si = 1
            for ss, ds in zip(st_shape, st_stride):
                if (d % ds == 0 and d // ds >= ss) or (s * d <= ds):
                    continue
                si = max(si, min(s, ss * ds // d))
            dst_inner.append(si)
            dst_outer.append(s // si)
            deps: List[Tuple[int, int]] = []
            for j, (ss, ds) in enumerate(zip(sv_shape, sv_stride)):
                if (d % ds == 0 and d // ds >= ss) or (s * d <= ds):
                    continue
                deps.append((j, ds))
                if j in src_readers:
                    src_readers[j].append(i)
                else:
                    src_readers[j] = [i]
            data.append((d, s, i, False, deps))

        for i, (s, d) in enumerate(zip(sv_shape, sv_stride)):
            si = 1
            for sd, dd in zip(dt_shape, dt_stride):
                if (d % dd == 0 and d // dd >= sd) or (s * d <= dd):
                    continue
                si = max(si, min(s, sd * dd // d))
            src_inner.append(si)
            src_outer.append(s // si)
            deps: List[Tuple[int, int]] = []
            for j, (sd, dd) in enumerate(zip(dv_shape, dv_stride)):
                if (d % dd == 0 and d // dd >= sd) or (s * d <= dd):
                    continue
                deps.append((j, dd))
                if j in dst_readers:
                    dst_readers[j].append(i)
                else:
                    dst_readers[j] = [i]
            data.append((d, s, i, True, deps))
        sorted_data = sorted(data, key=lambda x: -x[0])

        def update(d, s, i, olist_outer, olist_inner, ilist, deps):
            for j, di in deps:
                si = ilist[j]
                if si > 1:
                    olist_inner[i] = max(olist_inner[i], min(s, si * di // d))
                    olist_outer[i] = s // olist_inner[i]

        for d, s, i, is_src, deps in sorted_data:
            if is_src:
                update(d, s, i, src_outer, src_inner, dst_inner, deps)
                if i not in src_readers:
                    continue
                for j in src_readers[i]:
                    s = dv_shape[j]
                    d = dv_stride[j]
                    si = src_inner[i]
                    di = sv_stride[i]
                    if si > 1:
                        dst_inner[j] = max(dst_inner[j], min(s, si * di // d))
                        dst_outer[j] = s // dst_inner[j]
            else:
                update(d, s, i, dst_outer, dst_inner, src_inner, deps)
                if i not in dst_readers:
                    continue
                for j in dst_readers[i]:
                    s = sv_shape[j]
                    d = sv_stride[j]
                    si = dst_inner[i]
                    di = dv_stride[i]
                    if si > 1:
                        src_inner[j] = max(src_inner[j], min(s, si * di // d))
                        src_outer[j] = s // src_inner[j]

        src_inner_layout, src_outer_layout = self._calc_inner_outer_layout(src_outer, src_inner, sv_stride)
        dst_inner_layout, dst_outer_layout = self._calc_inner_outer_layout(dst_outer, dst_inner, dv_stride)

        sts_smem_layout = self._calc_smem_layout(shape, src, src_inner)
        lds_smem_layout = self._calc_smem_layout(shape, dst, dst_inner)
        sts_logical_layout = TensorLayout(tuple(src_inner), sv_stride)
        sts_layout = coalesce(composition(sts_smem_layout, sts_logical_layout))
        lds_logical_layout = TensorLayout(tuple(dst_inner), dv_stride)
        lds_layout = coalesce(composition(lds_smem_layout, lds_logical_layout))
        src_inner_layout, sts_layout = common_reshape(src_inner_layout, sts_layout)
        dst_inner_layout, lds_layout = common_reshape(dst_inner_layout, lds_layout)

        sts_layout = make_layout(coalesce(composition(sts_smem_layout, TensorLayout(st_shape, st_stride))), sts_layout)
        lds_layout = make_layout(coalesce(composition(lds_smem_layout, TensorLayout(dt_shape, dt_stride))), lds_layout)

        src_outer_layout, dst_outer_layout = common_reshape(src_outer_layout, dst_outer_layout)

        return (src_outer_layout, src_inner_layout, dst_outer_layout, dst_inner_layout, sts_layout, lds_layout)

    def emit(self, op: Rearrange, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        assert isinstance(src.layout, TiledTensorLayout) and isinstance(dst.layout, TiledTensorLayout)
        if op in self.op2exec_plan:
            (src_outer, src_inner, dst_outer, dst_inner, sts_layout, lds_layout) = self.op2exec_plan[op]
        else:
            shape = src.layout.shape()
            src_thr_layout = src.layout.thr_layout()
            src_val_layout = src.layout.val_layout()
            _shape = dst.layout.shape()
            dst_thr_layout = dst.layout.thr_layout()
            dst_val_layout = dst.layout.val_layout()
            if not shape == _shape:
                raise TypeError(f"Shape mismatch. (got:src({shape}),dst({_shape}))")
            src_tv_layout = make_layout(src_thr_layout, src_val_layout)
            dst_tv_layout = make_layout(dst_thr_layout, dst_val_layout)
            self.op2exec_plan[op] = (
                src_outer,
                src_inner,
                dst_outer,
                dst_inner,
                sts_layout,
                lds_layout,
            ) = self._schedule(shape, src_tv_layout, dst_tv_layout)
        smem_size = sts_layout.cosize()
        assert (
            smem_size == lds_layout.cosize()
        ), f"Schedule failed due to shared memory size mismatch. (got:sts({smem_size}), lds({lds_layout.cosize()}))"

        src_buf = src.buffer
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, (TensorType, PointerType))
        if isinstance(src_ty, TensorType):
            src_buf = ~src_buf[0]
            src_dtype = src_ty.dtype
        else:
            src_dtype = src_ty.base_type
        dst_buf = ~dst.buffer[0]
        dst_ty = infer_type(dst_buf)
        if not src_dtype == dst_ty.base_type:
            raise TypeError(f"Type mismatch. (got:src({src_dtype}), dst({dst_ty.base_type}))")

        smem_addr = var("smem", dst_ty)
        self.declare(smem_addr, self.get_smem_ptr(op, src_dtype, smem_size * src_dtype.nbytes))

        sts, sts_src_layout, sts_dst_layout = self._instruction_selection(
            Buffer(src_buf, None, src.dtype, src_inner, "register"),
            Buffer(smem_addr, None, src.dtype, sts_layout[1], "shared"),
        )
        lds, lds_src_layout, lds_dst_layout = self._instruction_selection(
            Buffer(smem_addr, None, dst.dtype, lds_layout[1], "shared"),
            Buffer(dst_buf, None, dst.dtype, dst_inner, "register"),
        )

        with self.for_grid(src_outer.shape) as coords:
            ## cond = [crd != 0 for crd in coords] if type(coords) is list else [coords != 0]
            ## with self.if_then(logical_or(*cond)):
            ##     self.append(syncthreads())
            self.append(syncthreads())
            src_addr_base = var("src_addr_base", dst_ty)
            smem_addr_base = var("smem_addr_base", dst_ty)
            self.declare(src_addr_base, src_buf + src_outer(coords, base=src.offset))
            self.declare(smem_addr_base, smem_addr + sts_layout[0](threadIdx.x))
            with self.for_grid(sts_src_layout[1].shape) as sts_coords:
                src_ptr = var("src_ptr", dst_ty)
                smem_ptr = var("smem_ptr", dst_ty)
                self.declare(src_ptr, src_addr_base + sts_src_layout[1](sts_coords))
                self.declare(smem_ptr, smem_addr_base + sts_dst_layout[1](sts_coords))
                self.append(sts(src_ptr, smem_ptr))
            self.append(syncthreads())
            dst_addr_base = var("dst_addr_base", dst_ty)
            self.declare(dst_addr_base, dst_buf + dst_outer(coords, base=dst.offset))
            self.assign(smem_addr_base, smem_addr + lds_layout[0](threadIdx.x))
            with self.for_grid(lds_src_layout[1].shape) as lds_coords:
                smem_ptr = var("smem_ptr", dst_ty)
                dst_ptr = var("dst_ptr", dst_ty)
                self.declare(smem_ptr, smem_addr_base + lds_src_layout[1](lds_coords))
                self.declare(dst_ptr, dst_addr_base + lds_dst_layout[1](lds_coords))
                self.append(lds(smem_ptr, dst_ptr))
