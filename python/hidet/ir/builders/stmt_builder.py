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
from typing import Union, Sequence, List, cast, Optional

from hidet.ir.type import BaseType
from hidet.ir.stmt import Stmt, ForStmt, IfStmt, EvaluateStmt, SeqStmt, LetStmt, ForMappingStmt, ForStmtAttr
from hidet.ir.stmt import DeclareStmt, BufferStoreStmt, AssignStmt, ReturnStmt, WhileStmt, BreakStmt, DeclareScope
from hidet.ir.stmt import AssertStmt
from hidet.ir.expr import Expr, Var, var, convert
from hidet.ir.mapping import RepeatTaskMapping
from hidet.ir.dtypes import int32
from hidet.ir.mapping import TaskMapping, repeat_map

ScopedStmt = Union[IfStmt, ForStmt, LetStmt, ForMappingStmt, WhileStmt]


class StmtScope:
    def __init__(self, sb, stmts: Union[ScopedStmt, Sequence[ScopedStmt]], ret=None):
        if not isinstance(stmts, Sequence):
            stmts = [stmts]

        assert all(isinstance(stmt, (IfStmt, ForStmt, LetStmt, ForMappingStmt, WhileStmt)) for stmt in stmts)

        self.sb: StmtBuilder = sb
        self.stmts: List[ScopedStmt] = list(stmts)
        self.ret = ret

    def __enter__(self):
        for stmt in self.stmts:
            self.sb.enter_body(stmt)
        return self.ret

    def __exit__(self, exc_type, exc_val, exc_tb):
        for _ in self.stmts:
            self.sb.exit_body()


class ElseIfScope:
    def __init__(self, sb, stmt: IfStmt):
        self.sb: StmtBuilder = sb
        self.stmt: IfStmt = stmt

    def __enter__(self):
        if not isinstance(self.sb.scope_stack[-1][-1], IfStmt):
            raise RuntimeError('else_if() must be called after if_then() or else_if()')

        # put the current one into the scope stack
        self.sb.scope_stack[-1].append(self.stmt)
        self.sb.scope_stack.append([])

    def __exit__(self, exc_type, exc_val, exc_tb):
        # exit the then body of the current if statement
        self.stmt.then_body = SeqStmt(self.sb.scope_stack.pop())
        self.sb.scope_stack[-1].pop()

        cur: IfStmt = self.sb.scope_stack[-1][-1]
        while cur.else_body is not None:
            if not isinstance(cur.else_body, IfStmt):
                raise RuntimeError('else_if() must be called after if_then() or else_if()')
            cur = cur.else_body

        cur.else_body = self.stmt


class OtherwiseScope:
    def __init__(self, sb):
        self.sb: StmtBuilder = sb

    def __enter__(self):
        if not isinstance(self.sb.scope_stack[-1][-1], IfStmt):
            raise RuntimeError('otherwise() must be called after if_then() or else_if()')
        self.sb.scope_stack.append([])

    def __exit__(self, exc_type, exc_val, exc_tb):
        else_body = SeqStmt(self.sb.scope_stack.pop())
        cur: IfStmt = self.sb.scope_stack[-1][-1]
        while cur.else_body is not None:
            if not isinstance(cur.else_body, IfStmt):
                raise RuntimeError('otherwise() must be called after if_then() or else_if()')
            cur = cur.else_body
        cur.else_body = else_body


class StmtBuilder:
    def __init__(self):
        # the structure of scope_stack:
        # [
        #    [...], # finished statements in outermost scope
        #    [...], # finished statements in the second outermost scope
        #    ...
        #    [...], # finished statements in the innermost scope
        # ]
        # when we exit a scope, it will be wrapped into a statement and append to outer scope
        self.scope_stack = [[]]

    def __iadd__(self, other: Union[Stmt, Expr, Sequence[Stmt]]):
        assert isinstance(other, (Stmt, Expr, list, tuple))
        self.append(other)
        return self

    @staticmethod
    def _name_index_vars(num_vars: int) -> List[str]:
        predefined_names = ['i', 'j', 'k', 'p', 'q', 'r', 's', 'u', 'v']
        if num_vars <= len(predefined_names):
            iter_names = predefined_names[:num_vars]
        else:
            iter_names = [f'i{idx}' for idx in range(num_vars)]
        return iter_names

    # singleton statements
    def declare(self, v: Var, init: Optional[Expr] = None, scope=None):
        self.append(DeclareStmt(v, init, scope=scope))
        return v

    def declare_var(
        self, name: str, tp: BaseType, init: Optional[Expr] = None, scope: Optional[DeclareScope] = None
    ) -> Var:
        v = var(name, tp)
        self.append(DeclareStmt(v, init=init, scope=scope))
        return v

    def buffer_store(self, buf: Expr, indices: Sequence[Union[Expr, int]], value: Expr):
        self.append(BufferStoreStmt(buf, convert(indices), value))

    def assign(self, dst: Var, value: Expr):
        self.append(AssignStmt(dst, value))

    def assertion(self, cond: Union[Expr, bool], msg: str) -> None:
        self.append(AssertStmt(cond, msg))

    def comment(self, comment_string: str, style: str = "//") -> None:
        from hidet.ir.primitives.debug import comment

        self.append(comment(comment_string, style=style))

    def brk(self):
        self.append(BreakStmt())

    # scope statements
    def let(self, v: Union[str, Var], value: Union[int, Expr]) -> StmtScope:
        if isinstance(v, str):
            v = var(v)
        return StmtScope(self, stmts=LetStmt(v, value), ret=v)

    def lets(self, bind_vars: Sequence[Union[str, Var]], values: Sequence[Union[int, Expr]]) -> StmtScope:
        assert len(bind_vars) == len(values)
        bind_vars = [var(v) if isinstance(v, str) else v for v in bind_vars]
        bind_values = [convert(value) for value in values]
        return StmtScope(self, stmts=LetStmt(bind_vars, bind_values, body=None), ret=bind_vars)

    def for_loop(self, v: Union[str, Var], extent: Union[int, Expr], attr: str = '.') -> StmtScope:
        if isinstance(v, str):
            v = var(v)
        return StmtScope(self, stmts=ForStmt(v, extent, attr=ForStmtAttr.parse(attr, num_loops=1)[0]), ret=v)

    def if_then(self, cond: Union[bool, Expr]) -> StmtScope:
        return StmtScope(self, stmts=IfStmt(cond), ret=None)

    def else_if(self, cond: Union[bool, Expr]) -> ElseIfScope:
        return ElseIfScope(self, IfStmt(cond))

    def otherwise(self) -> OtherwiseScope:
        return OtherwiseScope(self)

    def for_mapping(
        self,
        mapping: TaskMapping,
        iter_names: Optional[Sequence[str]] = None,
        worker: Optional[Union[Expr, int]] = None,
    ) -> StmtScope:
        if worker is None:
            if not isinstance(mapping, RepeatTaskMapping):
                raise ValueError('worker must be specified for non-repeat mapping')
            worker = int32.zero
        if iter_names is None:
            iter_names = self._name_index_vars(len(mapping.task_shape))
        iter_vars = [var(name) for name in iter_names]
        return StmtScope(self, stmts=ForMappingStmt(iter_vars, mapping, worker, cast(Stmt, None)), ret=iter_vars)

    def for_grid(self, shape: Sequence[Union[Expr, int]]) -> StmtScope:
        return self.for_mapping(mapping=repeat_map(shape), iter_names=self._name_index_vars(len(shape)), worker=0)

    def for_range(self, extent: Union[Expr, int], *, attr: Optional[Union[str, ForStmtAttr]] = None) -> StmtScope:
        iter_var = var("i")
        if isinstance(attr, str):
            attr = ForStmtAttr.parse(attr, num_loops=1)[0]
        else:
            attr = ForStmtAttr()
        return StmtScope(self, stmts=ForStmt(iter_var, extent, attr=attr), ret=iter_var)

    def while_loop(self, cond: Expr) -> StmtScope:
        return StmtScope(self, stmts=WhileStmt(cond, body=None), ret=None)

    def ret(self, value: Optional[Expr] = None):
        self.append(ReturnStmt(value))

    # utils
    def append(self, stmt: Union[Stmt, Expr, Sequence[Stmt], None]) -> None:
        if stmt is None:
            return
        if isinstance(stmt, (Stmt, Expr)):
            if isinstance(stmt, Expr):
                stmt = EvaluateStmt(stmt)
            self.scope_stack[-1].append(stmt)
        else:
            assert isinstance(stmt, (tuple, list))
            for s in stmt:
                self.append(s)

    def enter_body(self, stmt: Union[IfStmt, ForStmt, LetStmt, WhileStmt]):
        self.scope_stack[-1].append(stmt)
        self.scope_stack.append([])

    def exit_body(self):
        body = SeqStmt(self.scope_stack.pop())
        assert len(self.scope_stack) > 0
        last_stmt = self.scope_stack[-1][-1]
        if isinstance(last_stmt, (ForStmt, LetStmt, WhileStmt)):
            assert last_stmt.body is None
            last_stmt.body = body
        elif isinstance(last_stmt, IfStmt):
            if last_stmt.then_body is None:
                last_stmt.then_body = body
            else:
                assert last_stmt.else_body is None
                last_stmt.else_body = body
        elif isinstance(last_stmt, ForMappingStmt):
            last_stmt.body = body
        else:
            assert False

    def finish(self):
        assert len(self.scope_stack) == 1
        return SeqStmt(self.scope_stack.pop())
