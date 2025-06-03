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
from typing import List, Union, Dict
from hidet.ir.expr import Expr
from hidet.ir.type import PointerType, TensorType
from hidet.ir.tools import infer_type
from hidet.lang.cuda import threadIdx
from hidet.ir.dtypes import u64
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

from hidet.ir.cute.ops.partition import PartitionSrc, PartitionDst, PartitionA, PartitionB
from hidet.ir.cute import coalesce, composition, canonicalize_thread_value_layout, TensorLayout
from hidet.ir.cute.contexts import tid_in_groups

from hidet.utils import initialize
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

        if src.is_tma_buffer():
            assert src.scope.is_global()
            dst.tensor_maps = src.tensor_maps
            dst.coords = src.coords


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

        if src.is_tma_buffer():
            assert src.scope.is_global()
            dst.tensor_maps = src.tensor_maps
            dst.coords = src.coords


# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-descriptor-format
def encode_matrix_descriptor(x):
    return (x & 0x3FFFF) >> 0x4


# build smem matrix descriptor without smem address
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-descriptor-format
def make_wgmma_desc(lead_dim_offset, stride_dim_offset, layout_type) -> int:
    desc = 0
    desc |= encode_matrix_descriptor(lead_dim_offset) << 16
    desc |= encode_matrix_descriptor(stride_dim_offset) << 32
    desc |= layout_type << 62
    return desc


wgmma_descs: Dict[str, int] = {}


@initialize()
def register_wgmma_desc_template():
    wgmma_descs["Interleaved_N"] = make_wgmma_desc(128, 256, 0)
    wgmma_descs["SW32_N"] = make_wgmma_desc(1, 256, 3)
    wgmma_descs["SW64_N"] = make_wgmma_desc(1, 512, 2)
    wgmma_descs["SW128_N"] = make_wgmma_desc(1, 1024, 1)
    wgmma_descs["Interleaved_T"] = make_wgmma_desc(128, 256, 0)
    wgmma_descs["SW32_T"] = make_wgmma_desc(512, 256, 3)
    wgmma_descs["SW64_T"] = make_wgmma_desc(1024, 512, 2)
    wgmma_descs["SW128_T"] = make_wgmma_desc(2048, 1024, 1)


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
        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        if src.scope.is_register():
            dst.buffer = src_buf
            assert dst.offset is None
        elif src.scope.is_shared():
            _, a_tv = op.tiled_mma.a_tv_layout()
            assert "layout_type" in op.annotations
            layout_type = op.annotations["layout_type"]
            a_t, _ = canonicalize_thread_value_layout(a_tv)
            threads = a_t.size()
            warpgroup_layout = TensorLayout((128, threads // 128), (0, 128))
            thread_layout = composition(src.layout, a_t)
            thread_layout = coalesce(composition(thread_layout, warpgroup_layout))
            smem_addr = src_buf + thread_layout(tid, base=src_off)
            smem_addr = cvta_generic_to_shared(smem_addr)
            desc_template = self.auto_var(hint='desc', e=u64(wgmma_descs[layout_type]))
            matrix_start_addr = (smem_addr & 0x3FFFF) >> 4
            matrix_base_addr = ((smem_addr >> 0x7) & 0x7) << 49
            desc = self.auto_var(hint="desc", e=desc_template | matrix_start_addr | matrix_base_addr)
            dst.buffer = desc
            dst.offset = 0


@register_impl(PartitionB)
class PartitionBEmitter(OpEmitter):
    def emit(self, op: PartitionB, args: List[Union[Buffer, Expr]], output: Buffer):
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
        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        if src.scope.is_register():
            dst.buffer = src_buf
            assert dst.offset is None
        elif src.scope.is_shared():
            _, b_tv = op.tiled_mma.b_tv_layout()
            assert "layout_type" in op.annotations
            layout_type = op.annotations["layout_type"]
            b_t, _ = canonicalize_thread_value_layout(b_tv)
            threads = b_t.size()
            warpgroup_layout = TensorLayout((128, threads // 128), (0, 128))
            thread_layout = composition(src.layout, b_t)
            thread_layout = coalesce(composition(thread_layout, warpgroup_layout))
            smem_addr = src_buf + thread_layout(tid, base=src_off)
            smem_addr = cvta_generic_to_shared(smem_addr)
            desc_template = self.auto_var(hint='desc', e=u64(wgmma_descs[layout_type]))
            matrix_start_addr = (smem_addr & 0x3FFFF) >> 4
            matrix_base_addr = ((smem_addr >> 0x7) & 0x7) << 49
            desc = self.auto_var(hint="desc", e=desc_template | matrix_start_addr | matrix_base_addr)
            dst.buffer = desc
            dst.offset = 0
