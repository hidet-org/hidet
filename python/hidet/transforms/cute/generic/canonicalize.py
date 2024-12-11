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
"""
This module convert the original IR sequence to a canonical form. For example, consider the following IR sequence:
```python
a = make_tensor("float32", [10, 10])
b = relu(a * 2.0)
```
After applying this pass, the IR sequence will be transformed to:
```python
a = make_tensor("float32", [10, 10])
b1 = a * 2.0
b = relu(b1)
```

Basically, this pass will create a declare statement for each call operation in the CuTe IR sequence, which will
facilitate the later transformation passes.

Note:
Currently, all the passes in CuTe is designed to work with declare statements. But sometimes, we will apply the
declare_to_let pass to convert the declare statements to let statements first. This will break the functionality of
the passes in CuTe. So, we apply a `LetToDeclare` pass to revert the conversion. This is safe because we will apply
the declare_to_let pass again after applying all the passes in CuTe.

Classes:
    LetToDeclare: Transforms Let statements into Declare statements.
    Canonicalize: Canonicalizes various statements in the IR, including Declare, Assign, Let, and Evaluate statements.
    CanonicalizePass: Applies the Canonicalize transformation pass on a function.

Functions:
    canonicalize_pass: Creates and returns a CanonicalizePass instance.

Usage:
    The main entry point for using these transformations is the `canonicalize_pass` function, which returns a
    `CanonicalizePass` instance. This instance can then be applied to functions to perform the necessary
    transformations.
"""
from typing import List, Union

from hidet.ir.expr import Var, Expr, var
from hidet.ir.tools import infer_type
from hidet.ir.functors import IRRewriter

from hidet.ir.cute.expr import CallOp
from hidet.ir.cute.type import TiledTensorType

from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass
from hidet.ir.stmt import Stmt, AssignStmt, LetStmt, DeclareStmt, SeqStmt, EvaluateStmt


class LetToDeclare(IRRewriter):
    """
    A class to transform Let statements into Declare statements.

    Methods:
        visit_LetStmt(stmt: LetStmt) -> Union[Stmt, SeqStmt]:
            Processes Let statements and converts them to Declare statements.
    """

    def visit_LetStmt(self, stmt: LetStmt):
        """
        Processes Let statements and converts them to Declare statements.

        Args:
            stmt (LetStmt): The Let statement to process.

        Returns:
            Union[Stmt, SeqStmt]: The transformed statement.
        """
        if any(isinstance(v, CallOp) for v in stmt.bind_values):
            stmts: List[Stmt] = []
            for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
                stmts.append(DeclareStmt(bind_var, self.visit(bind_value)))
            stmts.append(self.visit(stmt.body))
            if len(stmts) == 1:
                return stmts[0]
            else:
                return SeqStmt(stmts)
        return super().visit_LetStmt(stmt)


class Canonicalize(IRRewriter):
    """
    A class to canonicalize various statements in a given function's intermediate representation (IR).

    Attributes:
        stmts (List[Stmt]): A list of statements to be processed.
        recursion_depth (int): The current depth of recursion.

    Methods:
        append_stmt(stmt: Union[Stmt, Expr]):
            Appends a statement or expression to the list of statements.

        flush_stmts() -> List[Stmt]:
            Flushes the list of statements and returns it.

        flatten_stmts(stmts: List[Stmt]) -> Union[Stmt, SeqStmt]:
            Flattens a list of statements into a single statement.

        visit_DeclareStmt(stmt: DeclareStmt) -> Union[Stmt, SeqStmt]:
            Processes and canonicalizes Declare statements.

        visit_AssignStmt(stmt: AssignStmt) -> Union[Stmt, SeqStmt]:
            Processes and canonicalizes Assign statements.

        visit_LetStmt(stmt: LetStmt) -> Union[Stmt, SeqStmt]:
            Processes and canonicalizes Let statements.

        visit_EvaluateStmt(stmt: EvaluateStmt) -> Union[Stmt, SeqStmt]:
            Processes and canonicalizes Evaluate statements.

        visit_CallOp(call: CallOp) -> Union[Expr, Var]:
            Processes and canonicalizes Call operations.
    """

    def __init__(self):
        super().__init__()
        self.stmts: List[Stmt] = []
        self.recursion_depth: int = 0

    def append_stmt(self, stmt: Union[Stmt, Expr]):
        if isinstance(stmt, Expr):
            stmt = EvaluateStmt(stmt)
        self.stmts.append(stmt)

    def flush_stmts(self):
        stmts = self.stmts
        self.stmts = []
        return stmts

    def flatten_stmts(self, stmts: List[Stmt]):
        if len(stmts) == 1:
            return stmts[0]
        else:
            return SeqStmt(stmts)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v = self.visit(stmt.var)
        init = self.visit(stmt.init) if stmt.init is not None else None
        if v is stmt.var and init is stmt.init:
            new_stmt = stmt
        else:
            new_stmt = DeclareStmt(v, init, stmt.is_static, stmt.scope)
        if isinstance(stmt.init, CallOp):
            self.append_stmt(new_stmt)
            return self.flatten_stmts(self.flush_stmts())
        else:
            return new_stmt

    def visit_AssignStmt(self, stmt: AssignStmt):
        v = self.visit(stmt.var)
        value = self.visit(stmt.value)
        if v is stmt.var and value is stmt.value:
            new_stmt = stmt
        else:
            new_stmt = AssignStmt(v, value)
        if isinstance(stmt.value, CallOp):
            self.append_stmt(new_stmt)
            return self.flatten_stmts(self.flush_stmts())
        else:
            return new_stmt

    def visit_LetStmt(self, stmt: LetStmt):
        stmts: List[Stmt] = []
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            bind_vars.append(self.visit(bind_var))
            bind_values.append(self.visit(bind_value))
            stmts.extend(self.flush_stmts())
        body = self.visit(stmt.body)
        from hidet.utils import same_list

        if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            stmts.append(LetStmt(bind_vars, bind_values, body))
            return self.flatten_stmts(stmts)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        e = self.visit(stmt.expr)
        if e is stmt.expr:
            new_stmt = stmt
        else:
            new_stmt = EvaluateStmt(e)
        if isinstance(stmt.expr, CallOp):
            self.append_stmt(new_stmt)
            return self.flatten_stmts(self.flush_stmts())
        else:
            return new_stmt

    def visit_CallOp(self, call: CallOp):
        args: List[Expr] = []
        for arg in call.op.args:
            if isinstance(arg, (tuple, list)):
                args.append(tuple(self.visit(v) for v in arg))
            elif isinstance(arg, CallOp):
                self.recursion_depth += 1
                args.append(self.visit(arg))
            else:
                assert isinstance(arg, (Var, Expr))
                args.append(self.visit(arg))
        op = call.op.reforward(args)
        if op is call.op:
            new_call = call
        else:
            new_call = op.make_call()
        if self.recursion_depth > 0:
            new_var_ty = infer_type(new_call)
            new_var = var(op.name, new_var_ty)
            assert isinstance(new_var_ty, TiledTensorType)
            self.append_stmt(DeclareStmt(new_var, new_call, False, new_var_ty.scope))
            self.recursion_depth -= 1
            return new_var

        return new_call


class CanonicalizePass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        return self.apply_transforms(func, [LetToDeclare(), Canonicalize()])


def canonicalize_pass() -> FunctionPass:
    return CanonicalizePass()
