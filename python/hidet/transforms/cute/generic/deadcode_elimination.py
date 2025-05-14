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
This pass eliminates the tile-level CuTe primitives that are not used in the program.
This pass could be optimized later. Currently, the algorithm may not be the most efficient and effective one.

The pass consists of three steps:
1. Perform tensor alias analysis to find the mapping between subtensors and tensors.
2. Dead code elimination:
    a. Find the users of each tensor and its subtensors.
        - Users of a tensor are the statements that use the tensor as an argument.
    b. Iterate over all the tensors and:
        - Determine if a tensor can be eliminated. A tensor can be eliminated when all its users satisfy
          the following conditions:
            - The user is not a Copy, an Atomic, or a CollectiveStore operation that writes to memory.
            - The user is not an Mma operation.
            - The user is a PartitionSrc, a PartitionDst, or a SubTensor operation.
        - If so, mark the statement that defines the tensor and its users as can_be_eliminated.
        - Remove the statements that marked as can_be_eliminated from the users of tensors.
        - Recursively apply the above steps to the input tensors of the eliminated operator until no
          statements can be marked as can_be_eliminated.
3. Remove all the statements marked as can_be_eliminated.

Remark: A tensor can be eliminated if its users are only PartitionSrc, PartitionDst, or SubTensor operations because:
In the programming model of Hexcute, a tensor is created, partitioned or sliced, and then used in actual computation
or data movement. If a tensor is only used in partitioning or slicing operations, it means that the tensor is not used
in the actual computation or data movement. Therefore, the tensor can be eliminated.
"""
from typing import List, Dict

from hidet.ir.expr import Var
from hidet.ir.tools import infer_type
from hidet.ir.functors import IRVisitor, IRRewriter

from hidet.ir.cute.expr import CallOp
from hidet.ir.cute.type import TiledTensorType

from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass
from hidet.ir.stmt import Stmt, LetStmt, AssignStmt, DeclareStmt, EvaluateStmt, SeqStmt

from hidet.ir.cute.ops import (
    PartitionSrc,
    PartitionDst,
    Copy,
    Mma,
    SubTensor,
    Atomic,
    TensorBase,
    MBarrierArrive,
    MBarrierWait,
)
from hidet.ir.cute.collective import CollectiveStore

from hidet.logging import logger, setConsoleLevel, DEBUG


class SubtensorMapper(IRVisitor):
    """
    This pass maps each subtensor to its corresponding tensor. It can handle tensors in global memory,
    shared memory, and register files, which is useful for the dead code elimination pass.

    In this pass, we consider the PartitionSrc, PartitionDst, and SubTensor create a subtensor from an
    original tensor. The read/write and compute of these subtensors are considered as the read/write and
    compute of the original tensor. This information is used in the dead code elimination pass.
    """

    def __init__(self):
        super().__init__()
        self.subtensor2subtensor: Dict[Var, Var] = {}
        self.subtensor2tensor: Dict[Var, Var] = {}

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if isinstance(stmt.init, CallOp):
            v = stmt.var
            call = stmt.init
            op = call.op
            if isinstance(op, (PartitionSrc, PartitionDst, SubTensor)):
                tensor = self.visit(op)
                self.subtensor2subtensor[v] = tensor

    def visit_Var(self, v: Var):
        if v not in self.subtensor2subtensor:
            return v
        else:
            return self.visit(self.subtensor2subtensor[v])

    def visit_PartitionSrc(self, op: PartitionSrc):
        return op.x

    def visit_PartitionDst(self, op: PartitionDst):
        return op.x

    def visit_SubTensor(self, op: SubTensor):
        return op.x

    def analyze(self, func: Function):
        self.visit(func)
        for v, _ in self.subtensor2subtensor.items():
            self.subtensor2tensor[v] = self.visit(v)

        return self.subtensor2tensor


class DeadcodeElimination(IRVisitor):
    def __init__(self, subtensor2tensor: Dict[Var, Var]):
        super().__init__()
        # auxiliary data structures
        self.var2users: Dict[Var, List[Stmt]] = {}
        self.var2defs: Dict[Var, List[DeclareStmt]] = {}
        self.let2eliminated_vars: Dict[LetStmt, List[Var]] = {}
        self.eliminated: List[Stmt] = []

        # results from subtensor mapper
        self.subtensor2tensor = subtensor2tensor

    def _is_cute_tile(self, v: Var):
        if not isinstance(v, Var):
            return False
        v_type = infer_type(v)
        return isinstance(v_type, TiledTensorType)

    def _is_register_tile(self, v: Var):
        if not isinstance(v, Var):
            return False
        v_type = infer_type(v)
        return isinstance(v_type, TiledTensorType) and v_type.scope.is_register()

    def _add_user(self, v: Var, stmt: Stmt):
        alias_var = self.visit(v)
        if alias_var not in self.var2users:
            self.var2users[alias_var] = [stmt]
        else:
            self.var2users[alias_var].append(stmt)

    def visit_Var(self, v: Var):
        if v in self.subtensor2tensor:
            return self.subtensor2tensor[v]
        else:
            return v

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            v = stmt.var
            assert v not in self.var2defs
            self.var2defs[v] = stmt
            op = call.op
            for arg in op.args:
                if self._is_cute_tile(arg):
                    self._add_user(arg, stmt)
            if v not in self.var2users:
                self.var2users[v] = []

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        if isinstance(stmt.expr, CallOp):
            call = stmt.expr
            op = call.op
            for arg in op.args:
                if self._is_cute_tile(arg):
                    self._add_user(arg, stmt)

    def visit_AssignStmt(self, stmt: AssignStmt):
        if isinstance(stmt.value, CallOp):
            call = stmt.value
            op = call.op
            self._add_user(stmt.var, stmt)
            for arg in op.args:
                if self._is_cute_tile(arg):
                    alias_var = self.visit(arg)
                    if arg not in self.var2users:
                        self.var2users[alias_var] = [stmt]
                    else:
                        if stmt not in self.var2users[arg]:
                            self.var2users[alias_var].append(stmt)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, CallOp):
                call = bind_value
                assert bind_var not in self.var2defs
                self.var2defs[bind_var] = stmt
                op = call.op
                for arg in op.args:
                    if self._is_cute_tile(arg):
                        self._add_user(arg, stmt)
        self.visit(stmt.body)

    def _extract_call_from_stmt(self, stmt: Stmt):
        if isinstance(stmt, EvaluateStmt):
            return stmt.expr
        else:
            assert isinstance(stmt, DeclareStmt)
            return stmt.init

    def _eliminate_stmt(self, stmt: Stmt, v: Var = None):
        if isinstance(stmt, LetStmt):
            # some of the bind_vars can be eliminated, while others cannot be eliminated.
            # we need to record the eliminated bind_vars for each LetStmt.
            if stmt not in self.let2eliminated_vars:
                self.let2eliminated_vars[stmt] = [v]
            else:
                self.let2eliminated_vars[stmt].append(v)

            for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
                if bind_var is v:
                    call = bind_value
                    op = call.op
                    for arg in op.args:
                        alias_var = self.visit(arg)
                        if self._is_cute_tile(arg):
                            if stmt in self.var2users[alias_var]:
                                self.var2users[alias_var].remove(stmt)
                                self._try_eliminate(arg)
        else:
            self.eliminated.append(stmt)
            call = self._extract_call_from_stmt(stmt)
            op = call.op
            for arg in op.args:
                alias_var = self.visit(arg)
                if self._is_cute_tile(arg):
                    if stmt in self.var2users[alias_var]:
                        self.var2users[alias_var].remove(stmt)
                        self._try_eliminate(arg)

    def _get_def_op(self, v: Var):
        assert v in self.var2defs
        def_stmt = self.var2defs[v]
        if isinstance(def_stmt, DeclareStmt):
            call = def_stmt.init
            assert isinstance(call, CallOp)
            return call.op
        else:
            assert isinstance(def_stmt, LetStmt)
            for bind_var, bind_value in zip(def_stmt.bind_vars, def_stmt.bind_values):
                if bind_var is v:
                    call = bind_value
                    assert isinstance(call, CallOp)
                    return call.op
        raise AssertionError(f"Cannot find the operator that declares {v} in {def_stmt}")

    def _try_eliminate(self, v: Var):
        alias_var = self.visit(v)
        users = self.var2users[alias_var]
        can_eliminate = True
        # Determine if a tensor can be eliminated.
        for stmt in users:
            if isinstance(stmt, EvaluateStmt):
                call = stmt.expr
                op = call.op
                if isinstance(op, (Copy, Atomic)):
                    v_type = infer_type(v)
                    # If a variable is used in a Copy or an Atomic operation, it cannot be eliminated
                    # when
                    # 1. The variable is a mask.
                    # 2. The variable is the source of the Copy operation.
                    # 3. The variable is the destination of the Copy operation and the variable
                    # is a global variable.
                    cond_mask = v is op.mask
                    cond_mbar = isinstance(op, Copy) and v is op.mbarrier
                    cond_src = alias_var is self.visit(op.src)
                    if alias_var is self.visit(op.dst) and v_type.scope.is_memory():
                        def_op = self._get_def_op(alias_var)
                        assert isinstance(def_op, TensorBase)
                        cond_dst = v_type.scope.is_global() or def_op.is_volatile()
                    else:
                        cond_dst = False
                    if cond_mask or cond_mbar or cond_src or cond_dst:
                        can_eliminate = False
                elif isinstance(op, (Mma, CollectiveStore)):
                    can_eliminate = False
                # Do not eliminate memory barrier operations.
                elif isinstance(op, (MBarrierArrive, MBarrierWait)):
                    can_eliminate = False
            elif isinstance(stmt, DeclareStmt):
                call = stmt.init
                op = call.op
                if not isinstance(op, (PartitionSrc, PartitionDst, SubTensor)):
                    can_eliminate = False
            else:
                can_eliminate = False

        if not can_eliminate:
            return

        def_stmt = self.var2defs[v]
        # dump debug infor
        # guard by the if statement to save the compilation time.
        level = logger.level
        if level == DEBUG:
            logger.debug("============= eliminate ==============")
            alias_var = self.visit(v)
            logger.debug(f"tensor: {alias_var}, subtensor: {v}, def_stmt: {def_stmt}")
            logger.debug("======================================")

        # Remove the statement that defines the tensor and its users.
        self._eliminate_stmt(def_stmt, v)
        logger.debug("============= remove user ==============")
        for user in users:
            if user not in self.eliminated:
                if level == DEBUG:
                    alias_var = self.visit(v)
                    logger.debug(f"tensor: {alias_var}, subtensor: {v}, user: {user}")
                self._eliminate_stmt(user)
        logger.debug("======================================")

    def eliminate(self, func: Function):
        # Step 1. Find the users of each tensor and its subtensors.
        self.visit(func)

        level = logger.level
        if level == DEBUG:
            logger.debug("============= var2users ==============")
            for v, users in self.var2users.items():
                alias_var = self.visit(v)
                logger.debug(f"begin user {alias_var} ==============")
                for stmt in users:
                    logger.debug(f"tensor: {alias_var}, subtensor: {v}, stmt: {stmt}")
                logger.debug(f"end user {alias_var} ==============")
            logger.debug("======================================")

        # Step 2. Eliminate the tensors that can be eliminated.
        for v, users in self.var2users.items():
            self._try_eliminate(v)
        return self.eliminated, self.let2eliminated_vars


class DeadcodeEliminationRewriter(IRRewriter):
    def __init__(self, stmts: List[Stmt], let2vars: Dict[LetStmt, List[Var]]):
        super().__init__()
        self.stmts = stmts
        self.let2vars = let2vars
        self.empty_stmt = SeqStmt([])

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if stmt not in self.stmts:
            return super().visit_DeclareStmt(stmt)
        return self.empty_stmt

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        if stmt not in self.stmts:
            return super().visit_EvaluateStmt(stmt)
        return self.empty_stmt

    def visit_AssignStmt(self, stmt: AssignStmt):
        if stmt not in self.stmts:
            return super().visit_AssignStmt(stmt)
        return self.empty_stmt

    def visit_LetStmt(self, stmt: LetStmt):
        if stmt not in self.let2vars:
            return super().visit_LetStmt(stmt)
        else:
            bind_vars = []
            bind_values = []
            for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
                if bind_var not in self.let2vars[stmt]:
                    bind_vars.append(self.visit(bind_var))
                    bind_values.append(self.visit(bind_value))
            body = self.visit(stmt.body)
            if len(bind_vars) > 0:
                return LetStmt(bind_vars, bind_values, body)
            return self.empty_stmt

    def visit_SeqStmt(self, stmt: SeqStmt):
        seq = []
        for s in stmt.seq:
            new_stmt = self.visit(s)
            if isinstance(new_stmt, SeqStmt):
                seq.extend(new_stmt.seq)
            else:
                seq.append(new_stmt)
        return SeqStmt(seq)


class DeadcodeEliminationPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        verbose = False

        orig_level = logger.level
        if verbose:
            setConsoleLevel(DEBUG)

        subtensor_mapper = SubtensorMapper()
        subtensor2tensor = subtensor_mapper.analyze(func)

        if verbose:
            logger.debug("============== tensor alias =================")
            for s, t in subtensor2tensor.items():
                logger.debug(f"subtensor: {s}, tensor: {t}")
            logger.debug("=============================================")

        deadcode_eliminator = DeadcodeElimination(subtensor2tensor)
        stmts, let2vars = deadcode_eliminator.eliminate(func)

        rewriter = DeadcodeEliminationRewriter(stmts, let2vars)
        func = rewriter.rewrite(func)
        if verbose:
            setConsoleLevel(orig_level)
        return func


def deadcode_elimination_pass() -> FunctionPass:
    return DeadcodeEliminationPass()
