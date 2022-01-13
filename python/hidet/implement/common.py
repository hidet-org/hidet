from typing import Mapping, Union
from hidet.ir import TensorInput, ScalarInput, ReduceCompute, TensorCompute
from hidet.ir.expr import *
from hidet.ir.dialects.lowlevel import Cast, Dereference
from hidet.ir.stmt import ForStmt, BufferStoreStmt, concat_stmts, AssignStmt, SeqStmt, StmtBuilder
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
        self.sb = StmtBuilder()
        self.input_map = input_map
        self.new_buffer_map = {}

    def expand(self, e):
        value = self.visit(e)
        return self.sb.finish(), value, self.new_buffer_map

    def visit_binary(self, e: BinaryOp):
        return e.__class__(self(e.a), self(e.b))

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
        return TensorSlice(self(e.base), e.indices, e.starts, e.ends)

    def visit_TensorElement(self, e: TensorElement):
        return TensorElement(self(e.base), [self(v) for v in e.indices])

    def visit_Call(self, e: Call):
        return Call(e.func_var, [self.visit(arg) for arg in e.args])

    def visit_Var(self, e: Var):
        return e

    def visit_Constant(self, e: Constant):
        return e

    def visit_TensorInput(self, e: TensorInput):
        return self.input_map[e]

    def visit_ScalarInput(self, e: ScalarInput):
        return self.input_map[e]

    def visit_TensorCompute(self, e: TensorCompute):
        # declare output buffer when needed
        if e in self.input_map:
            buf = self.input_map[e]
        else:
            buf = tensor_var(e.name, e.shape, dtype=infer_type(e.value))
            self.new_buffer_map[e] = buf

        # tensor compute loops
        for i in range(len(e.shape)):
            self.sb.append(ForStmt(e.axes[i], e.shape[i]))
            self.sb.enter_body()

        # at the inner-most loop body
        expr = self.visit(e.value)
        self.sb.append(BufferStoreStmt(buf, e.axes, expr))

        # exit loop scope
        for i in range(len(e.shape)):
            self.sb.exit_body()

        return buf

    def visit_ReduceCompute(self, e: ReduceCompute):
        # declare accumulator
        acc = scalar_var(e.name, infer_type(e.value))
        self.new_buffer_map[e] = acc

        # init accumulator
        self.sb.append(AssignStmt(acc, e.init_const()))

        # reduction loop
        assert len(e.shape) == 1
        self.sb.append(ForStmt(e.axis, e.shape[0]))

        with self.sb.for_body():
            expr = self.visit(e.value)
            self.sb.append(AssignStmt(acc, e.combine(acc, expr)))

        return acc

    def visit_Cast(self, e: Cast):
        return Cast(self(e.expr), e.target_type)

    def visit_Dereference(self, e: Dereference):
        return Dereference(self(e.expr))


def expand_loop(expr: Expr, input_map: Mapping[Union[ScalarInput, TensorInput, Expr], Var]):
    """
    Generate statements to calculate the expression.

    The expression may contain TensorCompute and ReduceCompute sub-expressions.
    After expand, the stmt will not have ScalarInput, TensorInput, TensorCompute and ReduceCompute anymore.

    The returned new_buffer_map is a mapping from ReduceCompute and TensorCompute sub-expressions to
    new allocated buffers used to finish the computation.

    For example, the following expr:
    compute([3, 3], (i, j) -> reduce_sum(A[i, k] * B[k, j], axis=k)) where k = axis(3)
    will be expanded to
    for i in range(3):
        for j in range(3):
            s = 0
            for k in range(3):
                s += A[i, k] * B[k, j]
            C[i, j] = s

    If C is in input_map, then the mapped var is used directly. Otherwise, a new tensor var is created to store the results
    and returned in new_buffer_map. We only reuse tensor in input_map.
    """
    expander = LoopExpander(input_map)
    stmt, value, new_buffer_map = expander.expand(expr)
    return stmt, value, new_buffer_map


