from hidet.ir import TensorInput, ScalarInput, ReduceCompute, TensorCompute
from hidet.ir.expr import *
from hidet.ir.dialects.lowlevel import Cast, Dereference
from hidet.ir.stmt import ForStmt, BufferStoreStmt, flatten, AssignStmt, SeqStmt
from hidet.ir.functors import ExprFunctor, infer_type


def merge_stmts(stmts):
    filtered = [s for s in stmts if s]
    if len(filtered) == 0:
        return None
    elif len(filtered) == 1:
        return filtered[0]
    else:
        return SeqStmt(filtered)


class LoopExpander(ExprFunctor):
    def __init__(self, input_map):
        super().__init__()
        self.input_map = input_map
        self.new_buffer_map = {}

    def expand(self, e):
        stmt, value = self.visit(e)
        return stmt, value, self.new_buffer_map

    def visit_binary(self, e: BinaryOp):
        sa, va = self.visit(e.a)
        sb, vb = self.visit(e.b)
        stmt = merge_stmts([sa, sb])
        return stmt, e.__class__(va, vb)

    def visit_Add(self, e: Add):
        return self.visit_binary(e)

    def visit_Sub(self, e: Sub):
        return self.visit_binary(e)

    def visit_Multiply(self, e: Multiply):
        return self.visit_binary(e)

    def visit_Div(self, e: Div):
        return self.visit_binary(e)

    def visit_Mod(self, e: Mod):
        return self.visit_binary(e)

    def visit_FloorDiv(self, e: FloorDiv):
        return self.visit_binary(e)

    def visit_LessThan(self, e: LessThan):
        return self.visit_binary(e)

    def visit_Equal(self, e: Equal):
        return self.visit_binary(e)

    def visit_TensorSlice(self, e: TensorSlice):
        base_stmt, base_expr = self.visit(e.base)
        return base_stmt, TensorSlice(base_expr, e.slices)

    def visit_TensorElement(self, e: TensorElement):
        stmts = []
        tensor_stmt, tensor_expr = self(e.base)
        if tensor_stmt:
            stmts.append(tensor_stmt)
        idx_exprs = []
        for idx in e.indices:
            idx_stmt, idx_expr = self.visit(idx)
            if idx_stmt:
                stmts.append(idx_stmt)
            idx_exprs.append(idx_expr)
        return merge_stmts(stmts), TensorElement(tensor_expr, idx_exprs)

    def visit_Call(self, e: Call):
        return None, Call(e.func_var, [self.visit(arg) for arg in e.args])

    def visit_Var(self, e: Var):
        return None, e

    def visit_Axis(self, e: Axis):
        return None, e

    def visit_Constant(self, e: Constant):
        return None, e

    def visit_TensorInput(self, e: TensorInput):
        return None, self.input_map[e]

    def visit_ScalarInput(self, e: ScalarInput):
        return None, self.input_map[e]

    def visit_TensorCompute(self, e: TensorCompute):
        if e in self.input_map:
            buf = self.input_map[e]
        else:
            buf = tensor_var(e.name, e.shape, dtype=infer_type(e.value))
            self.new_buffer_map[e] = buf
        stmts = []
        for i in range(len(e.shape)):
            stmts.append(ForStmt(e.axes[i]))
        stmt, expr = self.visit(e.value)
        sub_stmts = []
        if stmt is not None:
            sub_stmts.append(stmt)
        sub_stmts.append(BufferStoreStmt(buf, e.axes, expr))
        stmts.append(flatten(sub_stmts))
        return flatten(stmts), buf

    def visit_ReduceCompute(self, e: ReduceCompute):
        if e in self.input_map:
            acc = self.input_map[e]
        else:
            acc = scalar_var(e.name, infer_type(e.value))
            self.new_buffer_map[e] = acc

        seq_stmts = []
        seq_stmts.append(AssignStmt(acc, e.init_const()))

        stmts = []
        stmts.append(ForStmt(e.axis))
        stmt, expr = self.visit(e.value)
        if stmt:
            stmts.append(stmt)
        stmts.append(AssignStmt(acc, e.combine(acc, expr)))
        seq_stmts.append(flatten(stmts))
        return SeqStmt(seq_stmts), acc

    def visit_Cast(self, e: Cast):
        stmt, expr = self.visit(e.expr)
        return stmt, Cast(expr, e.target_type)

    def visit_Dereference(self, e: Dereference):
        stmt, expr = self.visit(e.expr)
        return stmt, Dereference(expr)


def expand_loop(expr, input_map):
    expander = LoopExpander(input_map)
    stmt, value, new_buffer_map = expander.expand(expr)
    return stmt, value, new_buffer_map


