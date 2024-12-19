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
from typing import List, Union, Optional
from hidet.ir.expr import Expr, Var, Constant, var, logical_and, cast
from hidet.ir.type import TensorType, PointerType
from hidet.ir.dtypes import u32, boolean
from hidet.ir.tools import infer_type
from hidet.lang.cuda import threadIdx

from hidet.ir.cute.ops import Copy, Mask, Atomic
from hidet.ir.cute.layout import TensorLayout, composition
from hidet.ir.cute.int_tuple import size, idx2crd

from .registry import OpEmitter, Buffer, register_impl


@register_impl(Mask)
class MaskEmitter(OpEmitter):
    def emit(self, op: Mask, args: List[Expr], output: Buffer):
        dst: Var = output.buffer
        shape, src_thrval_layout = op.tiled_copy.src_tv_layout()
        annotations = op.annotations
        assert annotations is not None
        rest_layout = annotations["rest_layout"]
        extents = rest_layout.shape
        index = TensorLayout(extents)
        nr_masks = size(extents)
        nr_regs = (nr_masks + u32.nbytes * 8 - 1) // (u32.nbytes * 8)

        base = var("base")
        self.declare(base, src_thrval_layout[0][0](threadIdx.x))
        with self.for_grid([nr_regs]) as i:
            self.assign(dst[i], 0)
        with self.for_grid(extents) as indices:
            idx = var("idx")
            mask_idx = var("mask_idx")
            bit = var("bit")
            crd = var("crd")
            pred = var("pred", boolean)
            self.declare(idx, index(indices))
            self.declare(mask_idx, (idx >> 5))
            self.declare(bit, (idx & 31))
            self.declare(crd, base + rest_layout(indices))
            cond = [e < v for v, e in zip(args, idx2crd(crd, shape))]
            self.declare(pred, logical_and(*cond))
            self.assign(dst[mask_idx], dst[mask_idx] | (cast(pred, u32) << bit))


@register_impl(Copy)
class CopyEmitter(OpEmitter):
    def emit(self, op: Copy, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        assert isinstance(args[0], Buffer)
        assert isinstance(args[1], Buffer)
        assert len(args) <= 2 or isinstance(args[2], Buffer)
        src: Buffer = args[0]
        dst: Buffer = args[1]
        mask: Optional[Buffer] = args[2] if len(args) >= 3 else None
        src_buf = src.buffer
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, PointerType)
        dst_buf = dst.buffer
        dst_ty = infer_type(dst_buf)
        assert isinstance(dst_ty, PointerType)
        shape, _ = op.tiled_copy.src_tv_layout()
        _shape, _ = op.tiled_copy.dst_tv_layout()
        assert shape == _shape
        var_names = ["src_addr", "dst_addr"]
        operand_tys = [src_ty, dst_ty]
        annotations = op.annotations
        assert len(annotations) > 0
        inst = annotations["inst"]
        src_layout = annotations["src_layout"]
        dst_layout = annotations["dst_layout"]
        attrs = op.attrs
        evict = attrs["evict"]

        extents = src_layout[1].shape
        if mask is not None:
            index = TensorLayout(extents)
        with self.for_grid(extents) as indices:
            src_addr, dst_addr = [var(vname, t) for vname, t in zip(var_names, operand_tys)]
            self.declare(src_addr, src.buffer + src_layout[1](indices, base=src.offset))
            self.declare(dst_addr, dst.buffer + dst_layout[1](indices, base=dst.offset))
            if mask is not None:
                idx = var("idx")
                mask_idx = var("mask_idx")
                bit = var("bit")
                pred = var("pred", boolean)
                self.declare(idx, index(indices))
                self.declare(mask_idx, (idx >> 5))
                self.declare(bit, (idx & 31))
                self.declare(pred, mask.buffer[mask_idx] & (Constant(1, u32) << bit))
            else:
                pred = None
            self.append(inst(src_addr, dst_addr, pred, evict=evict))


@register_impl(Atomic)
class AtomicEmitter(OpEmitter):
    def emit(self, op: Atomic, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        assert isinstance(args[0], Buffer)
        assert isinstance(args[1], Buffer)
        assert len(args) <= 2 or isinstance(args[2], Buffer)
        src: Buffer = args[0]
        dst: Buffer = args[1]
        mask: Optional[Buffer] = args[2] if len(args) >= 3 else None
        src_buf = src.buffer
        src_ty = infer_type(src_buf)
        if isinstance(src_ty, TensorType):
            src_buf = ~src_buf[0]
        else:
            assert isinstance(src_ty, PointerType)
        dst_buf = dst.buffer
        dst_ty = infer_type(dst_buf)
        assert isinstance(dst_ty, PointerType)
        src_shape = src.layout.shape()
        thr = src.layout.thr_layout()
        dst_shape = dst.layout.shape
        assert src_shape == dst_shape
        thr2mem = composition(dst.layout, thr)
        dst_offset = self.auto_var(hint="partition_dst", e=thr2mem(threadIdx.x, base=dst.offset))
        annotations = op.annotations
        assert len(annotations) > 0
        inst = annotations["inst"]
        src_layout = annotations["src_layout"]
        dst_layout = annotations["dst_layout"]

        extents = src_layout[1].shape
        if mask is not None:
            index = TensorLayout(extents)
        with self.for_grid(extents) as indices:
            src_addr = self.auto_var(hint="src_addr", e=src_buf + src_layout[1](indices, base=src.offset))
            dst_addr = self.auto_var(hint="dst_addr", e=dst_buf + dst_layout[1](indices, base=dst_offset))
            if mask is not None:
                idx = var("idx")
                mask_idx = var("mask_idx")
                bit = var("bit")
                pred = var("pred", boolean)
                self.declare(idx, index(indices))
                self.declare(mask_idx, (idx >> 5))
                self.declare(bit, (idx & 31))
                self.declare(pred, mask.buffer[mask_idx] & (Constant(1, u32) << bit))
            else:
                pred = None
            self.append(inst(src_addr, dst_addr, pred))
