from typing import Tuple, Dict
from collections import defaultdict
from hidet.ir.task import Grid, Host
from hidet.ir.func import *
from hidet.ir.stmt import *
from hidet.ir.expr import *
from hidet.ir.dialects.compute import ReduceCompute, TensorCompute, TensorInput, ScalarInput
from hidet.ir.functors import StmtExprFunctor, TypeFunctor, collect
from hidet.ir.dialects.lowlevel import VoidType, PointerType, Cast, Dereference
from hidet.utils.doc import Doc, NewLine, Text, join
from hidet.backend.call_graph import CallGraph


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


class CudaCodegen(StmtExprFunctor, TypeFunctor):
    def __init__(self):
        super().__init__()
        self.func_name_map = {}
        self.ir_module: Optional[IRModule] = None
        self.obj_name = {}
        self.name_id_clock = defaultdict(int)

    def get_obj_name(self, e: Expr, hint):
        if e in self.obj_name:
            return self.obj_name
        if hint:
            orig_name = hint
        else:
            alias = {
                'ScalarInput': 'scalar',
                'TensorInput': 'tensor',
                'Var': 'v',
                'IntVar': 'iv',
                'Axis': 'i'
            }
            class_name = str(e.__class__.__name__)
            orig_name = alias[class_name] if class_name in alias else class_name

        if orig_name in self.name_id_clock:
            name = orig_name
            while name in self.name_id_clock:
                self.name_id_clock[orig_name] += 1
                name = orig_name + '_' + str(self.name_id_clock[orig_name])
        else:
            self.name_id_clock[orig_name] = 0
            name = orig_name

        self.obj_name[e] = name
        return name

    @staticmethod
    def canonize_funcname(name: str):
        return name.replace('.', '_')

    def __call__(self, node) -> Doc:
        return self.visit(node)

    def visit(self, node):
        if isinstance(node, IRModule):
            return self.gen_module(node)
        elif isinstance(node, Function):
            return self.gen_func(node)
        elif isinstance(node, (Stmt, Expr)):
            return StmtExprFunctor.visit(self, node)
        elif isinstance(node, BaseType):
            return TypeFunctor.visit(self, node)
        else:
            raise ValueError()

    def gen_module(self, module: IRModule) -> Doc:
        self.ir_module = module
        doc = Doc()
        doc += Text('#include <cassert>') + NewLine()
        doc += Text('extern "C" {') + NewLine()

        call_graph = CallGraph(module)
        for node in call_graph.reversed_order:
            doc += self(node.func) + NewLine()

        doc += NewLine() + '}'
        return doc

    def gen_func(self, func: Function) -> Doc:
        self.name_id_clock.clear()
        self.obj_name.clear()

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
        write_params = get_write_params(func)
        for i in range(len(func.params)):
            param = func.params[i]
            if param in write_params and isinstance(param.type, ScalarType):
                ref = Text(' &')
            else:
                ref = Text(' ')
            param_docs.append(self(param.type) + ref + self(param))
        doc += join(param_docs, Text(', '))
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

    def visit_Equal(self, e: Equal):
        return Text('(') + self(e.a) + ' == ' + self(e.b) + ')'

    def visit_TensorSlice(self, e: TensorSlice):
        return Doc()

    def visit_TensorElement(self, e: TensorElement):
        return self(e.base) + '[' + join([self(idx) for idx in e.indices], ', ') + ']'

    def visit_Cast(self, e: Cast):
        return Text('(') + self.visit(e.target_type) + ')' + self(e.expr)

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
        param_doc = Text('(') + join([self(arg) for arg in e.args], Text(', ')) + ')'
        return func_name + launch_config + param_doc

    def visit_Var(self, e: Var):
        return Text(self.get_obj_name(e, e.hint))

    def visit_Axis(self, e: Axis):
        return Text(self.get_obj_name(e, e.hint))

    def visit_Constant(self, e: Constant):
        return Text(str(e.value))

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        return NewLine() + self(stmt.expr) + ';'

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        doc = NewLine()
        doc += self(stmt.buf)
        doc += Text('[') + join([self(idx) for idx in stmt.indices], ', ') + ']'
        doc += Text(' = ') + self(stmt.value) + ';'
        return doc

    def visit_AssignStmt(self, stmt: AssignStmt):
        return NewLine() + self(stmt.var) + ' = ' + self(stmt.value) + ';'

    def visit_LetStmt(self, stmt: LetStmt):
        doc = NewLine() + self(stmt.var.type) + ' ' + self.visit(stmt.var) + ' = ' + self.visit(stmt.value) + ';'
        doc += self(stmt.body)
        return doc

    def visit_ForStmt(self, stmt: ForStmt):
        var = stmt.loop_var
        init_doc = self(var.type) + ' ' + self(var) + ' = ' + self(var.min_value)
        if isinstance(var.min_value, Constant) and var.min_value.value == 0:
            cond_doc = self(var < var.extent)
        else:
            cond_doc = self(var < var.min_value + var.extent)
        update_doc = self(var) + ' = ' + self(var + 1)
        doc = NewLine() + Text('for (') + init_doc + '; ' + cond_doc + '; ' + update_doc + ') '
        doc += Text('{') + self(stmt.body).indent() + NewLine() + Text('} ')
        return doc

    def visit_IfStmt(self, stmt: IfStmt):
        doc = NewLine() + Text('if ') + self(stmt.cond) + ' '
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
        return Text('TensorType(') + self(t.scalar_type) + ', [' + join([self(s) for s in t.shape], ", ") + '], ' + t.scope.name + ')'

    def visit_PointerType(self, t: PointerType):
        return self(t.base_type) + Text('*')

    def visit_VoidType(self, t: VoidType):
        return Text('void')

    def visit_ScalarInput(self, e: ScalarInput):
        raise ValueError()

    def visit_TensorInput(self, e: TensorInput):
        raise ValueError()

    def visit_TensorCompute(self, e: TensorCompute):
        raise ValueError()

    def visit_ReduceCompute(self, e: ReduceCompute):
        raise ValueError()


def codegen(ir_module: IRModule) -> Tuple[str, Dict[str, str]]:
    gen = CudaCodegen()
    doc = gen(ir_module)
    return str(doc), gen.func_name_map


