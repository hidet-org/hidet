from typing import Optional, List, Set, Dict
import itertools
import operator
from hidet.ir.expr import Expr, Constant, Var, BinaryOp, Add, Sub, Multiply, Div, FloorDiv, Mod, is_const_int
from hidet.ir.expr import And, Or, Not, LessThan, LessEqual, Equal, is_one, is_zero, is_true, is_false, convert
from hidet.ir.type import ScalarType
from hidet.ir.stmt import Stmt, LetStmt, ForStmt, IfStmt, SeqStmt
from hidet.ir.func import IRModule, Function
from hidet.ir.node import Node
from hidet.ir.task import Grid, ThreadBlock, Warp, Thread, Host
from hidet.ir.functors import StmtExprRewriter, rewrite
from hidet.transforms import Pass
from hidet.ir.primitives import thread_idx, block_idx


class BoundInfo:
    _max_num_candidates = 1024
    _max_compute_iters = 10 ** 4

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
            self.candidates = candidates
        elif min_value is not None and max_value is not None and max_value - min_value <= BoundInfo._max_num_candidates:
            self.candidates = set(range(min_value, max_value + 1))
        else:
            self.min_value = min_value
            self.max_value = max_value

    @staticmethod
    def combine(lhs: 'BoundInfo', rhs: 'BoundInfo', op) -> 'BoundInfo':
        if lhs.value:
            lhs_candidates = [lhs.value]
        elif lhs.candidates:
            lhs_candidates = lhs.candidates
        else:
            lhs_candidates = None
        if rhs.value:
            rhs_candidates = [rhs.value]
        elif rhs.candidates:
            rhs_candidates = rhs.candidates
        else:
            rhs_candidates = None
        if lhs_candidates and rhs_candidates:
            if len(lhs_candidates) * len(rhs_candidates) <= BoundInfo._max_compute_iters:
                candidates = set()
                for lv in lhs_candidates:
                    for rv in rhs_candidates:
                        candidates.add(op(lv, rv))
                if len(candidates) == 1:
                    return BoundInfo(value=candidates.pop())
                else:
                    return BoundInfo(candidates=candidates)
        # fall back to use min/max value of lhs_candidates/rhs_candidates to infer
        if lhs_candidates is None:
            if lhs.min_value is None or lhs.max_value is None:
                return BoundInfo()
            else:
                lhs_candidates = {lhs.min_value, lhs.max_value}
        if rhs_candidates is None:
            if rhs.min_value is None or rhs.max_value is None:
                return BoundInfo()
            else:
                rhs_candidates = {rhs.min_value, rhs.max_value}
        if op is operator.add:
            return BoundInfo(min_value=min(lhs_candidates) + min(rhs_candidates),
                             max_value=max(lhs_candidates) + max(rhs_candidates))
        elif op is operator.sub:
            return BoundInfo(min_value=min(lhs_candidates) - max(rhs_candidates),
                             max_value=max(lhs_candidates) - min(rhs_candidates))
        elif op is operator.mul:
            candidates = [a * b for a, b in itertools.product([min(lhs_candidates), max(lhs_candidates)],
                                                              [min(rhs_candidates), max(rhs_candidates)])]
            return BoundInfo(min_value=min(candidates), max_value=max(candidates))
        elif op is operator.floordiv:
            if all(v > 0 for v in rhs_candidates):
                return BoundInfo(min_value=min(lhs_candidates) // max(rhs_candidates),
                                 max_value=max(lhs_candidates) // min(rhs_candidates))
            else:
                return BoundInfo()
        elif op is operator.mod:
            return BoundInfo()
        else:
            raise NotImplementedError()

    def possible_max_value(self):
        if self.value:
            return self.value
        elif self.candidates:
            return max(self.candidates)
        elif self.max_value:
            return self.max_value
        else:
            return None

    def possible_min_value(self):
        if self.value:
            return self.value
        elif self.candidates:
            return min(self.candidates)
        elif self.min_value:
            return self.min_value
        else:
            return None

    def is_one(self):
        return self.value == 1

    def is_zero(self):
        return self.value == 0

    def __add__(self, other):
        assert isinstance(other, BoundInfo)
        return self.combine(self, other, operator.add)

    def __sub__(self, other):
        assert isinstance(other, BoundInfo)
        return self.combine(self, other, operator.sub)

    def __mul__(self, other):
        assert isinstance(other, BoundInfo)
        return self.combine(self, other, operator.mul)

    def __floordiv__(self, other):
        assert isinstance(other, BoundInfo)
        return self.combine(self, other, operator.floordiv)

    def __mod__(self, other):
        assert isinstance(other, BoundInfo)
        return self.combine(self, other, operator.mod)


class BoundAwareSimplifier(StmtExprRewriter):
    op_dict = {
        Add: operator.add,
        Sub: operator.sub,
        Multiply: operator.mul,
        Div: operator.truediv,
        Mod: operator.mod,
        FloorDiv: operator.floordiv,
        LessThan: operator.lt,
        Equal: operator.eq
    }

    def __init__(self):
        super().__init__()
        self.bound: Dict[Expr, BoundInfo] = {}

    def __call__(self, node: Node):
        return self.visit(node)

    def get_bound(self, e):
        if e in self.bound:
            return self.bound[e]
        else:
            return BoundInfo()

    def visit(self, node: Node):
        if node in self.memo:
            return self.memo[node]
        if isinstance(node, (Stmt, Expr)):
            res = StmtExprRewriter.visit(self, node)
        elif isinstance(node, IRModule):
            res = self.visit_IRModule(node)
        elif isinstance(node, Function):
            res = self.visit_Function(node)
        else:
            raise ValueError(type(node))
        self.memo[node] = res
        return res

    def visit_IRModule(self, ir_module: IRModule) -> IRModule:
        updated = False
        funcs = {}
        for name, func in ir_module.functions.items():
            assert isinstance(func, Function), "Please resolve first"
            new_func = self(func)
            if new_func is not func:
                updated = True
            funcs[name] = new_func
        if updated:
            new_ir_module = IRModule(funcs, ir_module.task)
            new_ir_module.global_vars = ir_module.global_vars
            return new_ir_module
        else:
            return ir_module

    def visit_Function(self, func: Function) -> IRModule:
        worker = func.get_attr('worker')
        self.bound.clear()
        if isinstance(worker, Grid):
            self.bound[thread_idx()] = BoundInfo(min_value=0, max_value=int(worker.block_dim))
            self.bound[block_idx()] = BoundInfo(min_value=0, max_value=int(worker.grid_dim))
        elif isinstance(worker, ThreadBlock):
            self.bound[thread_idx()] = BoundInfo(min_value=0, max_value=int(worker.block_dim))
        elif isinstance(worker, Warp):
            # for warp worker, it can only get the lane id from thread idx, which is between 0...31
            self.bound[thread_idx()] = BoundInfo(min_value=0, max_value=32)
            self.bound[block_idx()] = BoundInfo(value=0)
        elif isinstance(worker, Thread):
            self.bound[thread_idx()] = BoundInfo(min_value=0, max_value=0)
            self.bound[block_idx()] = BoundInfo(value=0)
        elif isinstance(worker, Host):
            pass
        else:
            raise NotImplementedError()
        body = self(func.body)
        if body is func.body:
            return func
        else:
            return Function(func.name, func.params, body, func.ret_type, func.local_vars, func.attrs)

    def visit_LetStmt(self, stmt: LetStmt):
        var = self.visit_expr(stmt.var)
        value = self.visit_expr(stmt.value)
        if isinstance(value, Constant):
            body = rewrite(stmt.body, {var: value})
            return self(body)
        else:
            self.bound[var] = self.get_bound(stmt.value)
            body = self.visit(stmt.body)
            if var is stmt.var and value is stmt.value and body is stmt.body:
                return stmt
            else:
                return LetStmt(var, value, body)

    def visit_ForStmt(self, stmt: ForStmt):
        loop_var = self.visit_expr(stmt.loop_var)
        extent = self.visit_expr(stmt.extent)
        if is_one(extent):
            body = rewrite(stmt.body, {loop_var: extent})
            return self(body)
        else:
            self.bound[loop_var] = BoundInfo(min_value=0, max_value=self.get_bound(stmt.extent).possible_max_value())
            body = self.visit(stmt.body)
            if loop_var is stmt.loop_var and body is stmt.body:
                return stmt
            else:
                return ForStmt(loop_var, extent, stmt.unroll, body)

    def visit_IfStmt(self, stmt: IfStmt):
        cond = self.visit_expr(stmt.cond)
        then_body = self.visit(stmt.then_body)
        else_body = self.visit(stmt.else_body) if stmt.else_body else None
        if is_true(cond):
            return then_body
        elif is_false(cond):
            if else_body:
                return else_body
            else:
                return SeqStmt([])
        else:
            if cond is stmt.cond and then_body is stmt.then_body and else_body is stmt.else_body:
                return stmt
            else:
                return IfStmt(cond, then_body, else_body)

    def visit_Binary(self, e: BinaryOp):
        a = self(e.a)
        b = self(e.b)
        if isinstance(e, Add):
            if is_zero(a):
                self.bound[e] = self.get_bound(e.b)
                return b
            if is_zero(b):
                self.bound[e] = self.get_bound(e.a)
                return a
        elif isinstance(e, Sub):
            if is_zero(b):
                self.bound[e] = self.get_bound(e.a)
                return a
        elif isinstance(e, Multiply):
            if is_one(a):
                self.bound[e] = self.get_bound(e.b)
                return b
            if is_one(b):
                self.bound[e] = self.get_bound(e.a)
                return a
            if is_zero(a) or is_zero(b):
                self.bound[e] = BoundInfo(value=0)
                return convert(0)
        elif isinstance(e, Div):
            if is_one(b):
                self.bound[e] = self.get_bound(e.a)
                return a
        elif isinstance(e, Mod):
            if is_one(e.b):
                self.bound[e] = BoundInfo(value=0)
                return convert(0)
        elif isinstance(e, FloorDiv):
            if is_one(b):
                self.bound[e] = self.get_bound(e.a)
                return a
        elif isinstance(e, LessThan):
            a_max = self.get_bound(e.a).possible_max_value()
            b_min = self.get_bound(e.b).possible_min_value()
            if a_max and b_min and a_max < b_min:
                return convert(True)
        elif isinstance(e, LessEqual):
            a_max = self.get_bound(e.a).possible_max_value()
            b_min = self.get_bound(e.b).possible_min_value()
            if a_max and b_min and a_max <= b_min:
                return convert(True)
        elif isinstance(e, Equal):
            bound_a = self.get_bound(e.a)
            bound_b = self.get_bound(e.b)
            if bound_a.value and bound_b.value:
                return convert(bound_a.value == bound_b.value)
        elif isinstance(e, And):
            if is_false(a) or is_false(b):
                return convert(False)
            if is_true(a):
                return b
            if is_true(b):
                return a
        elif isinstance(e, Or):
            if is_true(a) or is_true(b):
                return convert(True)
            if is_false(a):
                return b
            if is_false(b):
                return a
        else:
            raise ValueError()

        if isinstance(a, Constant) and isinstance(b, Constant):
            if e.__class__ in self.op_dict:
                if a.dtype.name == 'int32' and b.dtype.name == 'int32' and isinstance(e, Div):
                    # the Div for int32 will use floordiv. Override the native behavior of python
                    res = a.value // b.value
                else:
                    res = convert(self.op_dict[e.__class__](a.value, b.value))
            elif isinstance(e, And):
                res = convert(a.value and b.value)
            elif isinstance(e, Or):
                res = convert(a.value or b.value)
            else:
                raise ValueError()
            if isinstance(res, Constant) and res.dtype.name == 'int32':
                self.bound[e] = BoundInfo(value=res.value)
            return res
        if isinstance(e, (Add, Sub, Multiply, FloorDiv, Mod)):
            self.bound[e] = BoundInfo.combine(self.get_bound(e.a), self.get_bound(e.b), self.op_dict[e.__class__])
            if self.bound[e].value is not None:
                return convert(self.bound[e].value)
        if a is e.a and b is e.b:
            return e
        return e.__class__(a, b)

    def visit_Add(self, e: Add):
        return self.visit_Binary(e)

    def visit_Sub(self, e: Sub):
        return self.visit_Binary(e)

    def visit_Multiply(self, e: Multiply):
        return self.visit_Binary(e)

    def visit_Div(self, e: Div):
        return self.visit_Binary(e)

    def visit_Mod(self, e: Mod):
        return self.visit_Binary(e)

    def visit_FloorDiv(self, e: FloorDiv):
        return self.visit_Binary(e)

    def visit_LessThan(self, e: LessThan):
        return self.visit_Binary(e)

    def visit_LessEqual(self, e: LessEqual):
        return self.visit_Binary(e)

    def visit_Equal(self, e: Equal):
        return self.visit_Binary(e)

    def visit_And(self, e: And):
        return self.visit_Binary(e)

    def visit_Or(self, e: Or):
        return self.visit_Binary(e)

    def visit_Not(self, e: Not):
        a = self(e.a)
        if a is e.a:
            return e
        else:
            return Not(a)

    def visit_Var(self, e: Var):
        return e

    def visit_Constant(self, e: Constant):
        if e.dtype.name == 'int32':
            self.bound[e] = BoundInfo(value=e.value)
        return e


class BoundAwareSimplifyPass(Pass):
    def __init__(self):
        super().__init__('bound_aware_simplify')

    def process_module(self, ir_module: IRModule) -> IRModule:
        simplifier = BoundAwareSimplifier()
        return simplifier(ir_module)


def bound_aware_simplify_pass() -> Pass:
    return BoundAwareSimplifyPass()
