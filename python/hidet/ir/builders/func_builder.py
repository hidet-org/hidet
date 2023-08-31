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
from typing import List, Dict, Optional

from hidet.ir.type import VoidType
from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.ir.stmt import Stmt

from .stmt_builder import StmtBuilder


class FunctionBuilder(StmtBuilder):
    def __init__(
        self,
        name: str,
        kind: str,
        label: str = "",
        ret_type=VoidType(),
        grid_dim=None,
        block_dim=None,
        dynamic_smem_bytes=None,
        min_blocks=None,
        attrs=None,
    ):
        super().__init__()
        self.name = name
        self.kind = kind
        self.params: List[Var] = []
        self.ret_type = ret_type
        self.func: Optional[Function] = None
        self.body: Optional[Stmt] = None
        self.attrs: Dict[str] = attrs if attrs else {}
        self.label = label

        if grid_dim is not None:
            self.attrs['cuda.grid_dim'] = grid_dim
        if block_dim is not None:
            self.attrs['cuda.block_dim'] = block_dim
        if dynamic_smem_bytes:
            self.attrs['cuda.dynamic_smem_bytes'] = dynamic_smem_bytes
        if min_blocks:
            self.attrs['cuda.min_blocks'] = min_blocks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finish_func()

    def extend_params(self, params: List[Var]):
        self.params.extend(params)

    def extend_attrs(self, new_attrs: Dict[str, object]):
        self.attrs.update(new_attrs)

    def set_body(self, body: Stmt):
        self.body = body

    def finish_func(self):
        # pylint: disable=import-outside-toplevel
        assert self.func is None
        if 'label' not in self.attrs:
            self.attrs['label'] = self.label
        if self.body is None:
            self.body = self.finish()
        if 'func_kind' not in self.attrs:
            self.attrs['func_kind'] = self.kind
        self.func = Function(
            self.name, kind=self.kind, params=self.params, body=self.body, ret_type=self.ret_type, attrs=self.attrs
        )

    def get(self) -> Function:
        assert self.func.body is not None
        return self.func
