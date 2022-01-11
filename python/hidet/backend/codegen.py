from typing import Tuple
from hidet.ir.dialects.pattern import ScalarExprPattern, TensorComputePattern, ReduceComputePattern
from hidet.ir.task import Grid, Host
from hidet.ir.func import *
from hidet.ir.stmt import *
from hidet.ir.expr import *
from hidet.ir.dialects.compute import ReduceCompute, TensorCompute, TensorInput, ScalarInput
from hidet.ir.functors import StmtExprFunctor, TypeFunctor, collect
from hidet.ir.dialects.lowlevel import VoidType, PointerType, Cast, Dereference, Address
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.ir.utils.call_graph import CallGraph
from hidet.utils.namer import Namer


class Codegen(StmtExprFunctor, TypeFunctor):
    def __init__(self):
        super().__init__()
        self.func_name_map = {}
        self.ir_module: Optional[IRModule] = None
        self.namer = Namer()

    @staticmethod
    def get_write_params(func: Function):
        params = func.params
        stmts = collect(func.body, (BufferStoreStmt, AssignStmt))
        write_params = []
        for param in params:
            for stmt in stmts:
                if isinstance(stmt, BufferStoreStmt):
                    if stmt.buf == param:
                        write_params.append(param)
                        break
                else:
                    assert isinstance(stmt, AssignStmt)
                    if stmt.var == param:
                        write_params.append(param)
                        break
        return write_params

    @staticmethod
    def canonize_funcname(name: str):
        return name.replace('.', '_')

    def __call__(self, node) -> Doc:
        return self.visit(node)

    def visit(self, node):
        if isinstance(node, IRModule):
            return self.visit_IRModule(node)
        elif isinstance(node, Function):
            return self.visit_Function(node)
        elif isinstance(node, (Stmt, Expr)):
            return StmtExprFunctor.visit(self, node)
        elif isinstance(node, BaseType):
            return TypeFunctor.visit(self, node)
        else:
            raise ValueError()

    def visit_IRModule(self, module: IRModule) -> Doc:
        self.ir_module = module
        doc = Doc()
        doc += Text('#include <cassert>') + NewLine()
        doc += Text('extern "C" {') + NewLine()

        call_graph = CallGraph(module)
        for node in call_graph.reversed_order:
            doc += self(node.func) + NewLine()

        doc += NewLine() + '}'
        return doc

    def visit_Function(self, func: Function) -> Doc:
        self.namer.clear()

        doc = NewLine()

        # ret
        worker = func.get_attr('worker')
        if isinstance(worker, Grid):
            doc += '__global__'
        elif isinstance(worker, Host):
            doc += '__host__'
        else:
            doc += '__device__ __forceinline__'
        doc += ' void'

        # func name
        canonized_func_name = self.canonize_funcname(func.name)
        doc += ' ' + canonized_func_name
        self.func_name_map[func.name] = canonized_func_name

        # parameters
        doc += '('
        param_docs = []
        write_params = self.get_write_params(func)
        for i in range(len(func.params)):
            param = func.params[i]
            if param in write_params and isinstance(param.type, ScalarType):
                ref = Text(' &')
            else:
                ref = Text(' ')
            param_docs.append(self(param.type) + ref + self(param))
        doc += doc_join(param_docs, Text(', '))
        doc += ') {'

        # locals
        for local_var in func.local_vars:
            doc += (NewLine() + self(local_var.type) + ' ' + self(local_var) + ';').indent()

        # body
        doc += self(func.body).indent()

        doc += NewLine() + '}'

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

    def visit_LessEqual(self, e: LessThan):
        return Text('(') + self(e.a) + ' <= ' + self(e.b) + ')'

    def visit_Equal(self, e: Equal):
        return Text('(') + self(e.a) + ' == ' + self(e.b) + ')'

    def visit_And(self, e: And):
        return Text('(') + self(e.a) + ' && ' + self(e.b) + ')'

    def visit_Or(self, e: Or):
        return Text('(') + self(e.a) + ' || ' + self(e.b) + ')'

    def visit_Not(self, e: Not):
        return Text('!') + self(e.a)

    def visit_TensorSlice(self, e: TensorSlice):
        slice_idx = 0
        base_doc = self(e.base)
        docs = []
        for idx in e.indices:
            if idx:
                docs.append(self(idx))
            else:
                start, end = e.starts[slice_idx], e.ends[slice_idx]
                docs.append(self(start) + ':' + self(end))
                slice_idx += 1
        return base_doc + '[' + doc_join(docs, ', ') + ']'

    def visit_TensorElement(self, e: TensorElement):
        return self(e.base) + '[' + doc_join([self(idx) for idx in e.indices], ', ') + ']'

    def visit_Cast(self, e: Cast):
        return Text('(') + self.visit(e.target_type) + ')' + self(e.expr)

    def visit_Address(self, e: Address):
        return Text('&') + self.visit(e.expr)

    def visit_Dereference(self, e: Dereference):
        return Text('*') + self(e.expr)

    def visit_Call(self, e: Call):
        func = self.ir_module.lookup(e.func_var.hint)
        worker = func.get_attr('worker')
        func_name = Text(self.canonize_funcname(e.func_var.hint))
        if isinstance(worker, Grid):
            block_dim = worker.block_dim
            grid_dim = worker.grid_dim
            launch_config = Text('<<<') + str(grid_dim) + ',' + str(block_dim) + Text('>>>')
        else:
            launch_config = []
        param_doc = Text('(') + doc_join([self(arg) for arg in e.args], Text(', ')) + ')'
        return func_name + launch_config + param_doc

    def visit_Var(self, e: Var):
        return Text(self.namer.get_name(e))

    def visit_Axis(self, e: Axis):
        return Text(self.namer.get_name(e))

    def visit_Constant(self, e: Constant):
        return Text(str(e.value))

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        return NewLine() + self(stmt.expr) + ';'

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        doc = NewLine()
        doc += self(stmt.buf)
        doc += Text('[') + doc_join([self(idx) for idx in stmt.indices], ', ') + ']'
        doc += Text(' = ') + self(stmt.value) + ';'
        return doc

    def visit_AssignStmt(self, stmt: AssignStmt):
        return NewLine() + self(stmt.var) + ' = ' + self(stmt.value) + ';'

    def visit_LetStmt(self, stmt: LetStmt):
        doc = NewLine() + self(stmt.var.type) + ' ' + self.visit(stmt.var) + ' = ' + self.visit(stmt.value) + ';'
        doc += self(stmt.body)
        return doc

    def visit_ForStmt(self, stmt: ForStmt):
        v = stmt.loop_var
        init_doc = self(v.type) + ' ' + self(v) + ' = ' + self(v.min_value)
        if isinstance(v.min_value, Constant) and v.min_value.value == 0:
            cond_doc = self(v < v.extent)
        else:
            cond_doc = self(v < v.min_value + v.extent)
        update_doc = self(v) + ' = ' + self(v + 1)
        doc = NewLine() + Text('for (') + init_doc + '; ' + cond_doc + '; ' + update_doc + ') '
        doc += Text('{') + self(stmt.body).indent() + NewLine() + Text('} ')
        return doc

    def visit_IfStmt(self, stmt: IfStmt):
        cond_doc = self(stmt.cond)
        if not(len(cond_doc.docs) > 0 and isinstance(cond_doc.docs[0], str) and cond_doc.docs[0].startswith('(')):
            cond_doc = Text('(') + cond_doc + ')'
        doc = NewLine() + Text('if ') + cond_doc + ' '
        doc += Text('{') + self(stmt.then_body).indent() + NewLine() + Text('} ')
        if stmt.else_body:
            doc += Text('else ')
            doc += Text('{') + self(stmt.else_body).indent() + NewLine() + Text('} ')
        return doc

    def visit_AssertStmt(self, stmt: AssertStmt):
        return NewLine() + Text('assert(((void)"') + stmt.msg + '", ' + self(stmt.cond) + '));'

    def visit_SeqStmt(self, stmt: SeqStmt):
        doc = Doc()
        for idx, s in enumerate(stmt.seq):
            doc += self(s)
        return doc

    def visit_ScalarType(self, t: ScalarType):
        scalar_type_map = {
            'int32': 'int32_t',
            'float32': 'float'
        }
        return Text(scalar_type_map[t.name])

    def visit_TensorType(self, t: TensorType):
        return Text('TensorType(') + self(t.scalar_type) + ', [' + doc_join([self(s) for s in t.shape], ", ") + '], ' + t.scope.name + ')'

    def visit_PointerType(self, t: PointerType):
        return self(t.base_type) + Text('*')

    def visit_VoidType(self, t: VoidType):
        return Text('void')

    # the following expressions should not remain to codegen
    def visit_ScalarInput(self, e: ScalarInput):
        raise ValueError()

    def visit_TensorInput(self, e: TensorInput):
        raise ValueError()

    def visit_TensorCompute(self, e: TensorCompute):
        raise ValueError()

    def visit_ReduceCompute(self, e: ReduceCompute):
        raise ValueError()

    def visit_AnyExpr(self, e: ReduceComputePattern):
        raise ValueError()

    def visit_ReduceComputePattern(self, e: ReduceComputePattern):
        raise ValueError()

    def visit_TensorComputePattern(self, e: TensorComputePattern):
        raise ValueError()

    def visit_ScalarExprPattern(self, e: ScalarExprPattern):
        raise ValueError()


def codegen(ir_module: IRModule) -> Tuple[str, Dict[str, str]]:
    gen = Codegen()
    doc = gen(ir_module)
    return str(doc), gen.func_name_map
