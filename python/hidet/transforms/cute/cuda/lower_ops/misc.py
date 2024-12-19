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
from hidet.ir.cute.ops.misc import Broadcast, Transpose

from .registry import OpEmitter, Buffer, register_impl


@register_impl(Broadcast)
class BroadcastEmitter(OpEmitter):
    def emit(self, op: Broadcast, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        assert src.scope.is_register()
        dst.buffer = src.buffer
        assert src.offset is None and dst.offset is None


@register_impl(Transpose)
class TransposeEmitter(OpEmitter):
    def emit(self, op: Transpose, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        assert src.scope.is_shared()
        dst.buffer = src.buffer
        assert src.offset is None and dst.offset is None
