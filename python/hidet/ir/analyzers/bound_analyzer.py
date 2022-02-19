import itertools
import operator
from collections import defaultdict
from typing import Optional, List, Set, Dict, Union, Mapping

from hidet.ir.expr import Expr, Var, Add, Sub, Multiply, FloorDiv, Mod, Constant
from hidet.ir.func import Function
from hidet.ir.functors import FuncStmtExprVisitor
from hidet.ir.primitives import thread_idx, block_idx
from hidet.ir.stmt import Stmt, LetStmt, ForStmt
from hidet.ir.task import Grid, ThreadBlock, Warp, Thread, Host


class BoundInfo:
    _max_num_candidates = 1024
    _max_compute_iters = 128 * 128

    def __init__(self, value=None, candidates=None, min_value=None, max_value=None):
        # three level of bound information:
        # 1. know its value
        # 2. know its candidates
        # 3. know it min_value and/or max_value
        # the specific one will hide the loose one, e.g., candidates are ignored when value is present.
        self.value: Optional[int] = None
        self.candidates: Optional[Set[int]] = None
        self.min_value: Optional[int] = None
        self.max_value: Optional[int] = None
        if value is not None:
            self.value = value
        elif candidates:
            if len(candidates) == 1:
                self.value = candidates[0]
            else:
                self.candidates = candidates
        elif min_value is not None and max_value is not None:
            if min_value == max_value:
                self.value = min_value
            elif max_value - min_value <= BoundInfo._max_num_candidates:
                self.candidates = set(range(min_value, max_value + 1))
            else:
                self.min_value = min_value
                self.max_value = max_value
        else:
            self.min_value = min_value
            self.max_value = max_value

    @staticmethod
    def combine(lhs: 'BoundInfo', rhs: 'BoundInfo', op) -> 'BoundInfo':
        if not lhs.has_determent_range() or not rhs.has_determent_range():
            return BoundInfo()

        lhs_candidates = lhs.candidate_set()
        rhs_candidates = rhs.candidate_set()
        if lhs_candidates and rhs_candidates and len(lhs_candidates) * len(rhs_candidates) <= BoundInfo._max_compute_iters:
            candidates = set()
            for lv in lhs_candidates:
                for rv in rhs_candidates:
                    candidates.add(op(lv, rv))
            if len(candidates) == 1:
                return BoundInfo(value=candidates.pop())
            else:
                return BoundInfo(candidates=candidates)
        else:
            # fall back to use min/max value of lhs_candidates/rhs_candidates to infer
            lhs_candidates = [lhs.possible_min_value(), lhs.possible_max_value()]
            rhs_candidates = [rhs.possible_min_value(), rhs.possible_max_value()]
            if op in [operator.add, operator.sub, operator.mul]:
                candidates = [op(a, b) for a, b in itertools.product([min(lhs_candidates), max(lhs_candidates)],
                                                                     [min(rhs_candidates), max(rhs_candidates)])]
                return BoundInfo(min_value=min(candidates), max_value=max(candidates))
            elif op is operator.floordiv:
                if all(v > 0 for v in rhs_candidates):
                    return BoundInfo(min_value=min(lhs_candidates) // max(rhs_candidates),
                                     max_value=max(lhs_candidates) // min(rhs_candidates))
                else:
                    return BoundInfo()
            elif op is operator.mod:
                if rhs.possible_max_value() is not None:
                    return BoundInfo(min_value=0, max_value=rhs.possible_max_value() - 1)
                else:
                    return BoundInfo()
            else:
                raise NotImplementedError()

    def candidate_set(self):
        if self.value is not None:
            return [self.value]
        elif self.candidates is not None:
            return self.candidates
        else:
            return None

    def has_determent_range(self) -> bool:
        if self.value is not None:
            return True
        if self.candidates is not None:
            return True
        if self.min_value is not None and self.max_value is not None:
            return True
        return False

    def possible_max_value(self):
        if self.value is not None:
            return self.value
        elif self.candidates:
            return max(self.candidates)
        elif self.max_value is not None:
            return self.max_value
        else:
            return None

    def possible_min_value(self):
        if self.value is not None:
            return self.value
        elif self.candidates:
            return min(self.candidates)
        elif self.min_value is not None:
            return self.min_value
        else:
            return None

    def is_one(self):
        return self.value == 1

    def is_zero(self):
        return self.value == 0

    def __add__(self, other):
        return self.combine(self, other, operator.add)

    def __sub__(self, other):
        return self.combine(self, other, operator.sub)

    def __mul__(self, other):
        return self.combine(self, other, operator.mul)

    def __floordiv__(self, other):
        return self.combine(self, other, operator.floordiv)

    def __mod__(self, other):
        return self.combine(self, other, operator.mod)

    def __lt__(self, other):
        lhs_max = self.possible_max_value()
        rhs_min = other.possible_min_value()
        return lhs_max is not None and rhs_min is not None and lhs_max < rhs_min

    def __le__(self, other):
        lhs_max = self.possible_max_value()
        rhs_min = other.possible_min_value()
        return lhs_max is not None and rhs_min is not None and lhs_max <= rhs_min

    def __str__(self):
        if self.value is not None:
            return str(self.value)
        elif self.candidates is not None:
            return str(len(self.candidates)) + ': ' + str(self.candidates)
        elif self.min_value or self.max_value:
            return f'[{self.min_value}:{self.max_value}]'
        else:
            return 'Any'


class BoundAnalyzer(FuncStmtExprVisitor):
    # we only infer bound based on variables from LetStmt and ForStmt, and the constants.
    # so the local variable with AssignStmt is not used infer bound.
    op_dict = {
        Add: operator.add,
        Sub: operator.sub,
        Multiply: operator.mul,
        FloorDiv: operator.floordiv,
        Mod: operator.mod
    }

    def __init__(self, var2bound: Dict[Expr, BoundInfo] = None):
        # please give the bound of external variable such as threadIdx.x using var2bound parameter
        super().__init__()
        self.bound: Dict[Expr, BoundInfo] = defaultdict(BoundInfo)
        if var2bound:
            self.bound.update(var2bound)

    def visit_Function(self, func: Function):
        worker = func.get_attr('worker')
        if isinstance(worker, Grid):
            self.bound[thread_idx()] = BoundInfo(min_value=0, max_value=int(worker.block_dim)-1)
            self.bound[block_idx()] = BoundInfo(min_value=0, max_value=int(worker.grid_dim)-1)
        elif isinstance(worker, ThreadBlock):
            self.bound[thread_idx()] = BoundInfo(min_value=0, max_value=int(worker.block_dim)-1)
            self.bound[block_idx()] = BoundInfo(value=0)
        elif isinstance(worker, Warp):
            # for warp worker, it can only get the lane id from thread idx, which is between 0...31
            self.bound[thread_idx()] = BoundInfo(min_value=0, max_value=1023)
            self.bound[block_idx()] = BoundInfo(value=0)
        elif isinstance(worker, Thread):
            self.bound[thread_idx()] = BoundInfo(min_value=0, max_value=1023)
            self.bound[block_idx()] = BoundInfo(value=0)
        elif isinstance(worker, Host):
            pass
        else:
            raise NotImplementedError()
        self.visit(func.body)

    def combine(self, e: Union[Add, Sub, Multiply, FloorDiv, Mod]):
        self.visit(e.a)
        self.visit(e.b)
        self.bound[e] = BoundAnalyzer.op_dict[e.__class__](self.bound[e.a], self.bound[e.b])

    def visit_Add(self, e: Add):
        self.combine(e)

    def visit_Sub(self, e: Sub):
        self.combine(e)

    def visit_Multiply(self, e: Multiply):
        self.combine(e)

    def visit_FloorDiv(self, e: FloorDiv):
        self.combine(e)

    def visit_Mod(self, e: Mod):
        self.combine(e)

    def visit_LetStmt(self, stmt: LetStmt):
        self.visit(stmt.value)
        self.bound[stmt.var] = self.bound[stmt.value]
        self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt):
        self.visit(stmt.extent)
        max_val = self.bound[stmt.extent].possible_max_value()
        if max_val is not None:
            max_val -= 1
        self.bound[stmt.loop_var] = BoundInfo(min_value=0, max_value=max_val)
        self.visit(stmt.body)

    def visit_Constant(self, e: Constant):
        if e.dtype.name == 'int32':
            self.bound[e] = BoundInfo(value=e.value)


def infer_bound(node: Union[Function, Stmt, Expr], var2bound: Optional[Mapping[Var, BoundInfo]] = None) -> Dict[Expr, BoundInfo]:
    visitor = BoundAnalyzer(var2bound)
    visitor.visit(node)
    return visitor.bound
