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

from hidet.ir.cute.ops.subtensor import SubTensor
from hidet.ir.cute import slice_and_offset

from .registry import OpEmitter, Buffer, register_impl


@register_impl(SubTensor)
class SubTensorEmitter(OpEmitter):
    def emit(self, op: SubTensor, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_buf = src.buffer
        src_off = src.offset
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, PointerType)
        _, offset = slice_and_offset(args[1], src.layout)
        dst.buffer = self.auto_var(hint=op.name, e=src_buf)
        from hidet.ir.dtypes import i32

        if src_off is None and offset == 0:
            dst.offset = i32(0)
        else:
            dst.offset = self.auto_var(hint=op.name, e=src_off + offset if src_off is not None else i32(offset))
