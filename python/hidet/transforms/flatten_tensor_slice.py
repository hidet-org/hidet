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
# pylint: disable=unused-variable
from hidet.ir.expr import TensorElement, TensorSlice
from hidet.ir.func import Function
from hidet.ir.functors import FuncStmtExprRewriter
from hidet.ir.stmt import BufferStoreStmt
from hidet.transforms import Pass


def concat_slices(lhs_indices, lhs_starts, lhs_ends, rhs_indices, rhs_starts=None, rhs_ends=None):
    if rhs_starts is None:
        rhs_starts = [None] * len(rhs_indices)
    if rhs_ends is None:
        rhs_ends = [None] * len(rhs_indices)
    assert len(lhs_indices) == len(lhs_starts) == len(lhs_ends)
    assert len(rhs_indices) == len(rhs_starts) == len(rhs_ends)
    indices = []
    starts = []
    ends = []
    i = 0
    for index, start, end in zip(lhs_indices, lhs_starts, lhs_ends):
        if index is not None:
            indices.append(index)
            starts.append(None)
            ends.append(None)
        else:
            assert i < len(rhs_indices)
            if rhs_indices[i] is not None:
                indices.append(start + rhs_indices[i] if start else rhs_indices[i])
                starts.append(None)
            elif rhs_starts[i] is not None:
                indices.append(None)
                starts.append(start + rhs_starts[i] if start else rhs_starts[i])
            else:
                indices.append(None)
                starts.append(None)
            # we ignore the end because we do not allow tensor-wise op.
            # end is only used for bound-checking, which is left in the future.
            ends.append(None)
            i += 1
    assert i == len(rhs_indices)
    return indices, starts, ends


class FlattenTensorSliceRewriter(FuncStmtExprRewriter):
    # eliminate all TensorSlice
    # (A[:, 3])[2] will be converted to A[2, 3] and the slice op A[:, 3] will be eliminated.
    def visit_TensorSlice(self, e: TensorSlice):
        base = self.visit(e.base)
        if isinstance(base, TensorSlice):
            e_indices = [self.visit(i) if i else None for i in e.indices]
            e_starts = [self.visit(s) if s else None for s in e.starts]
            e_ends = [self.visit(e) if e else None for e in e.ends]
            indices, starts, ends = concat_slices(base.indices, base.starts, base.ends, e_indices, e_starts, e_ends)
            return TensorSlice(base.base, indices, starts, ends)
        else:
            return FuncStmtExprRewriter.visit_TensorSlice(self, e)

    def visit_TensorElement(self, e: TensorElement):
        base = self.visit(e.base)
        if isinstance(base, TensorSlice):
            e_indices = [self.visit(idx) for idx in e.indices]
            indices, starts, ends = concat_slices(base.indices, base.starts, base.ends, e_indices)
            assert not any(idx is None for idx in indices)
            return TensorElement(base.base, indices, e.protected)
        else:
            return FuncStmtExprRewriter.visit_TensorElement(self, e)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        base = self.visit(stmt.buf)
        stmt_indices = [self.visit(idx) for idx in stmt.indices]
        if isinstance(base, TensorSlice):
            indices, starts, ends = concat_slices(base.indices, base.starts, base.ends, stmt_indices)
            assert not any(idx is None for idx in indices)
            return BufferStoreStmt(base.base, indices, self.visit(stmt.value), stmt.protected)
        else:
            return FuncStmtExprRewriter.visit_BufferStoreStmt(self, stmt)


class FlattenTensorSlicePass(Pass):
    def process_func(self, func: Function) -> Function:
        flatten_slice = FlattenTensorSliceRewriter()
        return flatten_slice(func)


def flatten_tensor_slice_pass() -> Pass:
    return FlattenTensorSlicePass()
