from hidet.ir.dialects.lowlevel import Reference, Address, ReferenceType, TensorPointerType, Dereference, Cast, VoidType, PointerType
from hidet.ir.expr import Call, TensorElement, Not, Or, And, Constant, Var, Let, Equal, LessThan, FloorDiv, Mod, Div, Multiply, Sub, Add, TensorType, ScalarType, Expr
from hidet.ir.functors import ExprFunctor, TypeFunctor
from hidet.ir.type import TypeNode
from hidet.ir.utils.hash_sum import HashSum


class ExprHash(ExprFunctor, TypeFunctor):
    def __init__(self):
        super().__init__()

    def visit(self, e):
        if isinstance(e, (str, float, int)):
            return HashSum(e)
        elif isinstance(e, (list, tuple)):
            return HashSum((self(v) for v in e))
        elif isinstance(e, Expr):
            return ExprFunctor.visit(self, e)
        else:
            assert isinstance(e, TypeNode)
            return TypeFunctor.visit(self, e)

    def hash(self, expr):
        return self(expr)

    def visit_Var(self, e: Var):
        return HashSum(e) + 'v'

    def visit_Constant(self, e: Constant):
        return HashSum(e.value) + self(e.dtype) + 'c'

    def visit_Add(self, e: Add):
        return (self(e.a) & self(e.b)) + '+'

    def visit_Sub(self, e: Sub):
        return self(e.a) + self(e.b) + '-'

    def visit_Multiply(self, e: Multiply):
        return (self(e.a) & self(e.b)) + '*'

    def visit_Div(self, e: Div):
        return self(e.a) + self(e.b) + '/'

    def visit_Mod(self, e: Mod):
        return self(e.a) + self(e.b) + '%'

    def visit_FloorDiv(self, e: FloorDiv):
        return self(e.a) + self(e.b) + '//'

    def visit_LessThan(self, e: LessThan):
        return self(e.a) + self(e.b) + '<<'

    def visit_LessEqual(self, e: LessThan):
        return self(e.a) + self(e.b) + '<='

    def visit_Equal(self, e: Equal):
        return (self(e.a) & self(e.b)) + '=='

    def visit_And(self, e: And):
        return (self(e.a) & self(e.b)) + '&&'

    def visit_Or(self, e: Or):
        return (self(e.a) & self(e.b)) + '||'

    def visit_Not(self, e: Not):
        return self(e.a) + '!'

    def visit_TensorElement(self, e: TensorElement):
        return self(e.base) + self(e.indices) + '[]'

    def visit_Cast(self, e: Cast):
        return self(e.expr) + self(e.target_type) + 'cast'

    def visit_Dereference(self, e: Dereference):
        return self(e.expr) + 'deref'

    def visit_Address(self, e: Address):
        return self(e.expr) + 'addr'

    def visit_Reference(self, e: Reference):
        return self(e.expr) + 'ref'

    def visit_Call(self, e: Call):
        return self(e.func_var) + self(e.args) + 'call'

    def visit_Let(self, e: Let):
        return self(e.var) + self(e.value) + self(e.body) + 'let'

    def visit_ScalarType(self, t: ScalarType):
        return self(t.name) + 'dtype'

    def visit_TensorType(self, t: TensorType):
        return self(t.scalar_type) + self(t.scope.name) + self(t.shape) + 'tensor_type'

    def visit_PointerType(self, t: PointerType):
        return self(t.base_type) + 'pointer_type'

    def visit_TensorPointerType(self, t: TensorPointerType):
        return self(t.tensor_type) + 'tensor_pointer_type'

    def visit_ReferenceType(self, t: ReferenceType):
        return self(t.base_type) + 'reference_type'

    def visit_VoidType(self, t: VoidType):
        return self('void_type')

