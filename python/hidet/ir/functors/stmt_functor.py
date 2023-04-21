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
# pylint: disable=bad-staticmethod-argument
from typing import List

from hidet.ir.node import Node
from hidet.ir.expr import Expr, Var
from hidet.ir.stmt import (
    EvaluateStmt,
    DeclareStmt,
    BufferStoreStmt,
    AssignStmt,
    LetStmt,
    ForStmt,
    ForMappingStmt,
    SeqStmt,
)
from hidet.ir.stmt import WhileStmt, BreakStmt, ContinueStmt, IfStmt, ReturnStmt, AsmStmt, AssertStmt, BlackBoxStmt
from hidet.ir.stmt import LaunchKernelStmt
from hidet.utils import same_list

from .base_functor import BaseFunctor, BaseVisitor, BaseRewriter


class StmtFunctor(BaseFunctor):
    def visit_dispatch(self, node: Node):
        if isinstance(node, EvaluateStmt):
            return self.visit_EvaluateStmt(node)
        elif isinstance(node, DeclareStmt):
            return self.visit_DeclareStmt(node)
        elif isinstance(node, BufferStoreStmt):
            return self.visit_BufferStoreStmt(node)
        elif isinstance(node, AssignStmt):
            return self.visit_AssignStmt(node)
        elif isinstance(node, LetStmt):
            return self.visit_LetStmt(node)
        elif isinstance(node, ForStmt):
            return self.visit_ForStmt(node)
        elif isinstance(node, ForMappingStmt):
            return self.visit_ForTaskStmt(node)
        elif isinstance(node, WhileStmt):
            return self.visit_WhileStmt(node)
        elif isinstance(node, BreakStmt):
            return self.visit_BreakStmt(node)
        elif isinstance(node, ContinueStmt):
            return self.visit_ContinueStmt(node)
        elif isinstance(node, IfStmt):
            return self.visit_IfStmt(node)
        elif isinstance(node, ReturnStmt):
            return self.visit_ReturnStmt(node)
        elif isinstance(node, AsmStmt):
            return self.visit_AsmStmt(node)
        elif isinstance(node, LaunchKernelStmt):
            return self.visit_LaunchKernelStmt(node)
        elif isinstance(node, AssertStmt):
            return self.visit_AssertStmt(node)
        elif isinstance(node, BlackBoxStmt):
            return self.visit_BlackBoxStmt(node)
        elif isinstance(node, SeqStmt):
            return self.visit_SeqStmt(node)
        else:
            return NotImplemented

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        raise NotImplementedError()

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        raise NotImplementedError()

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        raise NotImplementedError()

    def visit_AssignStmt(self, stmt: AssignStmt):
        raise NotImplementedError()

    def visit_LetStmt(self, stmt: LetStmt):
        raise NotImplementedError()

    def visit_ForStmt(self, stmt: ForStmt):
        raise NotImplementedError()

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        raise NotImplementedError()

    def visit_WhileStmt(self, stmt: WhileStmt):
        raise NotImplementedError()

    def visit_BreakStmt(self, stmt: BreakStmt):
        raise NotImplementedError()

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        raise NotImplementedError()

    def visit_IfStmt(self, stmt: IfStmt):
        raise NotImplementedError()

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        raise NotImplementedError()

    def visit_AssertStmt(self, stmt: AssertStmt):
        raise NotImplementedError()

    def visit_AsmStmt(self, stmt: AsmStmt):
        raise NotImplementedError()

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        raise NotImplementedError()

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        raise NotImplementedError()

    def visit_SeqStmt(self, stmt: SeqStmt):
        raise NotImplementedError()


class StmtVisitor(StmtFunctor, BaseVisitor):
    def visit_DeclareStmt(self, stmt: DeclareStmt):
        self.visit(stmt.var)
        if stmt.init is not None:
            self.visit(stmt.init)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        self.visit(stmt.expr)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        self.visit(stmt.buf)
        self.visit(stmt.value)
        for idx in stmt.indices:
            self.visit(idx)

    def visit_AssignStmt(self, stmt: AssignStmt):
        self.visit(stmt.var)
        self.visit(stmt.value)

    def visit_LetStmt(self, stmt: LetStmt):
        for _, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.visit(bind_value)
        self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt):
        self.visit(stmt.extent)
        self.visit(stmt.body)

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        for loop_var in stmt.loop_vars:
            self.visit(loop_var)
        self.visit(stmt.worker)
        self.visit(stmt.body)

    def visit_WhileStmt(self, stmt: WhileStmt):
        self.visit(stmt.cond)
        self.visit(stmt.body)

    def visit_BreakStmt(self, stmt: BreakStmt):
        pass

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        pass

    def visit_IfStmt(self, stmt: IfStmt):
        self.visit(stmt.cond)
        self.visit(stmt.then_body)
        if stmt.else_body:
            self.visit(stmt.else_body)

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        self.visit(stmt.ret_value)

    def visit_AssertStmt(self, stmt: AssertStmt):
        self.visit(stmt.cond)

    def visit_AsmStmt(self, stmt: AsmStmt):
        for expr in stmt.input_exprs:
            self.visit(expr)
        for expr in stmt.output_exprs:
            self.visit(expr)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        self.visit(stmt.func_var)
        for arg in stmt.args:
            self.visit(arg)
        for dim in stmt.grid_dim:
            self.visit(dim)
        for dim in stmt.block_dim:
            self.visit(dim)
        self.visit(stmt.shared_mem_bytes)

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        for expr in stmt.exprs:
            self.visit(expr)

    def visit_SeqStmt(self, stmt: SeqStmt):
        for s in stmt.seq:
            self.visit(s)


class StmtRewriter(StmtFunctor, BaseRewriter):
    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v = self.visit(stmt.var)
        init = self.visit(stmt.init) if stmt.init is not None else None
        if v is stmt.var and init is stmt.init:
            return stmt
        else:
            return DeclareStmt(v, init, stmt.is_static)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        e = self.visit(stmt.expr)
        if e is stmt.expr:
            return stmt
        else:
            return EvaluateStmt(e)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        buf = self.visit(stmt.buf)
        indices = [self.visit(e) for e in stmt.indices]
        value = self.visit(stmt.value)
        if buf is stmt.buf and all(a is b for a, b in zip(indices, stmt.indices)) and value is stmt.value:
            return stmt
        else:
            return BufferStoreStmt(buf, indices, value, stmt.protected)

    def visit_AssignStmt(self, stmt: AssignStmt):
        v = self.visit(stmt.var)
        value = self.visit(stmt.value)
        if v is stmt.var and value is stmt.value:
            return stmt
        else:
            return AssignStmt(v, value)

    def visit_LetStmt(self, stmt: LetStmt):
        bind_values = [self.visit(bind_value) for bind_value in stmt.bind_values]
        body = self.visit(stmt.body)
        if same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            return LetStmt(stmt.bind_vars, bind_values, body)

    def visit_ForStmt(self, stmt: ForStmt):
        loop_var = stmt.loop_var
        extent = self.visit(stmt.extent)
        body = self.visit(stmt.body)
        if loop_var is stmt.loop_var and extent is stmt.extent and body is stmt.body:
            return stmt
        else:
            return ForStmt(loop_var, extent, body=body, attr=stmt.attr)

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        loop_vars: List[Expr] = [self.visit(v) for v in stmt.loop_vars]
        # todo: visit expressions in task mapping
        worker = self.visit(stmt.worker)
        body = self.visit(stmt.body)
        if same_list(loop_vars, stmt.loop_vars) and worker is stmt.worker and body is stmt.body:
            return stmt
        else:
            assert all(isinstance(v, Var) for v in loop_vars)
            asserted_loop_vars: List[Var] = [v for v in loop_vars if isinstance(v, Var)]  # avoid IDE warning
            return ForMappingStmt(loop_vars=asserted_loop_vars, mapping=stmt.mapping, worker=worker, body=body)

    def visit_WhileStmt(self, stmt: WhileStmt):
        cond = self.visit(stmt.cond)
        body = self.visit(stmt.body)
        if cond is stmt.cond and body is stmt.body:
            return stmt
        else:
            return WhileStmt(cond, body)

    def visit_BreakStmt(self, stmt: BreakStmt):
        return stmt

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        return stmt

    def visit_IfStmt(self, stmt: IfStmt):
        cond = self.visit(stmt.cond)
        then_body = self.visit(stmt.then_body)
        else_body = self.visit(stmt.else_body) if stmt.else_body else None
        if cond is stmt.cond and then_body is stmt.then_body and else_body is stmt.else_body:
            return stmt
        else:
            return IfStmt(cond, then_body, else_body)

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        ret_value = self.visit(stmt.ret_value) if stmt.ret_value is not None else None
        if ret_value is stmt.ret_value:
            return stmt
        else:
            return ReturnStmt(ret_value)

    def visit_AssertStmt(self, stmt: AssertStmt):
        cond = self.visit(stmt.cond)
        if cond is stmt.cond:
            return stmt
        else:
            return AssertStmt(cond, stmt.msg)

    def visit_AsmStmt(self, stmt: AsmStmt):
        input_exprs = [self.visit(e) for e in stmt.input_exprs]
        output_exprs = [self.visit(e) for e in stmt.output_exprs]
        if same_list(input_exprs, stmt.input_exprs) and same_list(output_exprs, stmt.output_exprs):
            return stmt
        else:
            return AsmStmt(
                stmt.template_string,
                list(zip(stmt.output_labels, output_exprs)),
                list(zip(stmt.input_labels, input_exprs)),
                stmt.is_volatile,
            )

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        func_var = self.visit(stmt.func_var)
        args = [self.visit(e) for e in stmt.args]
        grid_dim = (self.visit(stmt.grid_dim[0]), self.visit(stmt.grid_dim[1]), self.visit(stmt.grid_dim[2]))
        block_dim = (self.visit(stmt.block_dim[0]), self.visit(stmt.block_dim[1]), self.visit(stmt.block_dim[2]))
        shared_mem_bytes = self.visit(stmt.shared_mem_bytes)
        if same_list(
            [func_var, *args, *grid_dim, *block_dim, shared_mem_bytes],
            [stmt.func_var, *stmt.args, *stmt.grid_dim, *stmt.block_dim, stmt.shared_mem_bytes],
        ):
            return stmt
        else:
            return LaunchKernelStmt(func_var, args, grid_dim, block_dim, shared_mem_bytes)

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        exprs = [self.visit(e) for e in stmt.exprs]
        if same_list(exprs, stmt.exprs):
            return stmt
        else:
            return BlackBoxStmt(stmt.template_string, *exprs)

    def visit_SeqStmt(self, stmt: SeqStmt):
        seq = []
        for s in stmt.seq:
            seq.append(self.visit(s))
        if all(a is b for a, b in zip(seq, stmt.seq)):
            return stmt
        else:
            return SeqStmt(seq)
