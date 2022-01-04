from collections import defaultdict
from hidet.ir.func import IRModule, Function
from hidet.ir.type import ScalarType, TensorType, Type, VoidType, PointerType
from hidet.ir.expr import Constant, Axis, Var, Call, TensorElement, TensorSlice, Add, Multiply, Expr, LessThan, FloorDiv, Mod, Equal, Dereference, Cast, Div, Sub
from hidet.ir.stmt import SeqStmt, IfStmt, ForStmt, LetStmt, AssignStmt, BufferStoreStmt, EvaluateStmt, Stmt, AssertStmt
from hidet.core.compute import ReduceCompute, TensorCompute, TensorInput, ScalarInput
from hidet.utils.doc import Doc, NewLine, Text, join

from .base import StmtExprFunctor, TypeFunctor


class IRPrinter(StmtExprFunctor, TypeFunctor):
    def __init__(self):
        super().__init__()
        self.obj_name = {}
        self.class_id_clock = defaultdict(int)

    def __call__(self, node):
        return self.visit(node)

    def get_obj_name(self, e: Expr):
        if e in self.obj_name:
            return self.obj_name
        alias = {
            'ScalarInput': 'scalar',
            'TensorInput': 'tensor',
            'Var': 'v',
            'IntVar': 'iv',
            'Axis': 'i'
        }
        self.class_id_clock[e.__class__] += 1
        id = self.class_id_clock[e.__class__]
        class_name = str(e.__class__.__name__)
        class_name = alias[class_name] if class_name in alias else class_name
        name = class_name + '_' + str(id)
        self.obj_name[e] = name
        return name

    def visit(self, obj):
        if isinstance(obj, Type):
            return TypeFunctor.visit(self, obj)
        elif isinstance(obj, Function):
            return self.visit_Function(obj)
        elif isinstance(obj, IRModule):
            return self.visit_IRModule(obj)
        else:
            return StmtExprFunctor.visit(self, obj)

    def visit_Function(self, func: Function):
        self.obj_name = {}
        self.class_id_clock = defaultdict(int)
        doc = Doc()

        # parameters
        doc += 'fn('
        param_docs = []
        for i in range(len(func.params)):
            param = func.params[i]
            param_docs.append([NewLine(), self(param), ': ', self(param.type)])
        doc += join(param_docs, Text(', '))
        doc += ')'
        doc = doc.indent(6)

        # locals
        for local_var in func.local_vars:
            doc += (NewLine() + Text('declare ') + self(local_var) + Text(': ') + self(local_var.type)).indent(4)

        # body
        doc += self(func.body).indent(4)

        return doc

    def visit_IRModule(self, ir_module: IRModule):
        doc = Doc()
        for name, func in ir_module.functions.items():
            doc += ['def ', name, ' ', self(func), NewLine(), NewLine()]
        return doc

    def visit_Add(self, e: Add):
        return Text('(') + self(e.a) + ' + ' + self(e.b) + ')'

    def visit_Sub(self, e: Sub):
        return Text('(') + self(e.a) + ' - ' + self(e.b) + ')'

    def visit_Multiply(self, e: Multiply):
        return Text('(') + self(e.a) + ' * ' + self(e.b) + ')'

    def visit_Div(self, e: Div):
        return Text('(') + self(e.a) + ' / ' + self(e.b) + ')'

    def visit_Mod(self, e: Mod):
        return Text('(') + self(e.a) + ' % ' + self(e.b) + ')'

    def visit_FloorDiv(self, e: FloorDiv):
        return Text('(') + self(e.a) + ' / ' + self(e.b) + ')'

    def visit_LessThan(self, e: LessThan):
        return Text('(') + self(e.a) + ' < ' + self(e.b) + ')'

    def visit_Equal(self, e: Equal):
        return Text('(') + self(e.a) + ' == ' + self(e.b) + ')'

    def visit_TensorSlice(self, e: TensorSlice):
        return Doc()

    def visit_TensorElement(self, e: TensorElement):
        return self(e.base) + '[' + join([self(idx) for idx in e.indices], ', ') + ']'

    def visit_Call(self, e: Call):
        return Text(e.func_var.hint) + '(' + join([self(arg) for arg in e.args], Text(', ')) + ')'

    def visit_Cast(self, e: Cast):
        return Text('cast(') + self(e.target_type) + ', ' + self(e.expr) + ')'

    def visit_Dereference(self, e: Dereference):
        return Text('*') + self(e.expr)

    def visit_Var(self, e: Var):
        if e.hint:
            return Text(e.hint)
        return Text(self.get_obj_name(e))

    def visit_Axis(self, e: Axis):
        if e.hint:
            return Text(e.hint)
        return Text(self.get_obj_name(e))

    def visit_Constant(self, e: Constant):
        return Text(str(e.value))

    def visit_ScalarInput(self, e: ScalarInput):
        return Text(self.get_obj_name(e))

    def visit_TensorInput(self, e: TensorInput):
        return Text(self.get_obj_name(e))

    def visit_TensorCompute(self, e: TensorCompute):
        return Doc()

    def visit_ReduceCompute(self, e: ReduceCompute):
        return Doc()

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        return NewLine() + self(stmt.expr)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        doc = NewLine()
        doc += self(stmt.buf)
        doc += Text('[') + join([self(idx) for idx in stmt.indices], ', ') + ']'
        doc += Text(' = ') + self(stmt.value)
        return doc

    def visit_AssignStmt(self, stmt: AssignStmt):
        return NewLine() + self(stmt.var) + ' = ' + self(stmt.value)

    def visit_LetStmt(self, stmt: LetStmt):
        doc = NewLine() + Text('let ') + self.visit(stmt.var) + ' = ' + self.visit(stmt.value)
        doc += self.visit(stmt.body)
        return doc

    def visit_ForStmt(self, stmt: ForStmt):
        var = stmt.loop_var
        if isinstance(var.min_value, Constant) and var.min_value.value == 0:
            rng = Text('range(') + self(var.extent) + ')'
        else:
            rng = Text('range(') + self(var.min_value) + ', ' + self(var.min_value) + ' + ' + self(var.extent) + ')'
        doc = NewLine() + Text('for ') + self(stmt.loop_var) + ' in ' + rng
        doc += self(stmt.body).indent(4)
        return doc

    def visit_IfStmt(self, stmt: IfStmt):
        doc = NewLine() + Text('if ') + self(stmt.cond)
        doc += self(stmt.then_body).indent(4)
        if stmt.else_body:
            doc += Text('else')
            doc += self(stmt.else_body).indent(4)
        return doc

    def visit_AssertStmt(self, stmt: AssertStmt):
        return NewLine() + 'assert(' + self(stmt.cond) + ', ' + stmt.msg + ')'

    def visit_SeqStmt(self, stmt: SeqStmt):
        doc = Doc()
        for idx, s in enumerate(stmt.seq):
            doc += self(s)
        return doc

    def visit_ScalarType(self, t: ScalarType):
        return Text('DataType({})'.format(t.name))

    def visit_TensorType(self, t: TensorType):
        return Text('TensorType(') + self(t.scalar_type) + ', [' + join([self(s) for s in t.shape], ", ") + '], ' + t.scope.name + ')'

    def visit_PointerType(self, t: PointerType):
        return Text('PointerType(') + self(t.base_type) + ')'

    def visit_VoidType(self, t: VoidType):
        return Text('VoidType')


def astext(obj) -> str:
    if isinstance(obj, (Expr, Stmt, Type, Function, IRModule)):
        printer = IRPrinter()
        return str(printer(obj))
    else:
        raise ValueError()

