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
from hidet.ir.type import PointerType
from hidet.ir.tools import infer_type
from hidet.ir.stmt import DeclareScope
from hidet.lang.cuda import threadIdx

from hidet.ir.cute.ops.partition import PartitionSrc, PartitionDst
from hidet.ir.cute.layout import composition

from .registry import OpEmitter, Buffer, register_impl


@register_impl(PartitionSrc)
class PartitionSrcEmitter(OpEmitter):
    def emit(self, op: PartitionSrc, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_var = src.var
        src_ty = infer_type(src_var)
        assert isinstance(src_ty, PointerType)
        _, src_thrval_layout = op.tiled_copy.src_tv_layout()
        if src.scope == DeclareScope.Register:
            self.assign(dst.var, src_var)
        else:
            thr_layout = composition(src.layout, src_thrval_layout[0][0])
            self.assign(dst.var, src_var + thr_layout(threadIdx.x))


@register_impl(PartitionDst)
class PartitionDstEmitter(OpEmitter):
    def emit(self, op: PartitionDst, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_var = src.var
        src_ty = infer_type(src_var)
        assert isinstance(src_ty, PointerType)
        _, dst_thrval_layout = op.tiled_copy.dst_tv_layout()
        if src.scope == DeclareScope.Register:
            self.assign(dst.var, src_var)
        else:
            thr_layout = composition(src.layout, dst_thrval_layout[0][0])
            self.assign(dst.var, src_var + thr_layout(threadIdx.x))
