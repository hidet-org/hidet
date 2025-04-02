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
from hidet.ir.expr import Expr
from hidet.ir.type import PointerType, TensorType
from hidet.ir.tools import infer_type
from hidet.lang.cuda import threadIdx

from hidet.ir.cute.ops.partition import PartitionSrc, PartitionDst, PartitionA
from hidet.ir.cute import composition
from hidet.ir.cute.contexts import tid_in_groups

from .registry import OpEmitter, Buffer, register_impl


@register_impl(PartitionSrc)
class PartitionSrcEmitter(OpEmitter):
    def emit(self, op: PartitionSrc, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_buf = src.buffer
        src_off = src.offset
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, (PointerType, TensorType))
        if isinstance(src_ty, TensorType):
            indices = [0] * len(src_ty.shape)
            src_buf = ~src_buf[indices]
        _, src_thrval_layout = op.tiled_copy.src_tv_layout()
        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        if src.scope.is_register():
            dst.buffer = src_buf
            assert dst.offset is None
        else:
            thr_layout = composition(src.layout, src_thrval_layout[0][0])
            dst.buffer = src_buf
            dst.offset = self.auto_var(hint=op.name, e=thr_layout(tid, base=src_off))


@register_impl(PartitionDst)
class PartitionDstEmitter(OpEmitter):
    def emit(self, op: PartitionDst, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_buf = src.buffer
        src_off = src.offset
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, (PointerType, TensorType))
        if isinstance(src_ty, TensorType):
            indices = [0] * len(src_ty.shape)
            src_buf = ~src_buf[indices]
        _, dst_thrval_layout = op.tiled_copy.dst_tv_layout()
        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        if src.scope.is_register():
            dst.buffer = src_buf
            assert dst.offset is None
        else:
            thr_layout = composition(src.layout, dst_thrval_layout[0][0])
            dst.buffer = src_buf
            dst.offset = self.auto_var(hint=op.name, e=thr_layout(tid, base=src_off))


@register_impl(PartitionA)
class PartitionAEmitter(OpEmitter):
    def emit(self, op: PartitionA, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_buf = src.buffer
        src_off = src.offset
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, (PointerType, TensorType))
        if isinstance(src_ty, TensorType):
            indices = [0] * len(src_ty.shape)
            src_buf = ~src_buf[indices]
        _, a_thrval_layout = op.tiled_mma.a_tv_layout()

        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        if src.scope.is_register():
            dst.buffer = src_buf
            assert dst.offset is None
        else:
            thr_layout = composition(src.layout, a_thrval_layout[0][0])
            dst.buffer = src_buf
            dst.offset = self.auto_var(hint=op.name, e=thr_layout(tid, base=src_off))
