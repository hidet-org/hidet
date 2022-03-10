from typing import Tuple
from hidet.ir.dialects.pattern import ScalarExprPattern, TensorComputePattern, ReduceComputePattern
from hidet.ir.task import Grid, Host
from hidet.ir.func import *
from hidet.ir.stmt import *
from hidet.ir.expr import *
from hidet.ir.dialects.compute import ReduceCompute, TensorCompute, TensorInput, ScalarInput
from hidet.ir.functors import StmtExprFunctor, TypeFunctor, collect, simplify
from hidet.ir.dialects.lowlevel import VoidType, PointerType, Cast, Dereference, Address, ReferenceType, Reference, TensorPointerType
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.ir.utils.call_graph import CallGraph
from hidet.utils.namer import Namer
from hidet.ir.primitives import is_primitive_function, get_primitive_function


class Codegen(StmtExprFunctor, TypeFunctor):
    def __init__(self):
        super().__init__()
        self.func_name_map = {}
        self.ir_module: Optional[IRModule] = None
        self.namer = Namer()

    @staticmethod
    def canonize_funcname(name: str):
        return name.replace('.', '_')

    def param_declare(self, v: Var):
        v_type = v.type
        name_doc = self(v)
        if isinstance(v_type, ScalarType):
            dtype_doc = self(v_type)
            return dtype_doc + ' ' + name_doc
        elif isinstance(v_type, (TensorPointerType, PointerType)):
            dtype = v_type.base_type if isinstance(v_type, PointerType) else v_type.tensor_type.scalar_type
            base_type_doc = self(dtype)
            return base_type_doc + ' *' + name_doc
        elif isinstance(v_type, ReferenceType):
            if isinstance(v_type.base_type, ScalarType):
                base_type_doc = self(v_type.base_type)
                return base_type_doc + ' &' + name_doc
            else:
                raise NotImplementedError()
        elif isinstance(v_type, TensorType):
            dtype_doc = self(v_type.scalar_type)
            name_doc = self(v)
            shape_doc = Doc()
            for s in v_type.shape:
                shape_doc += '[' + self(s) + ']'
            return dtype_doc + ' ' + name_doc + shape_doc
        else:
            raise ValueError()

    def local_var_declare(self, v: Var):
        v_type = v.type
        if isinstance(v_type, ScalarType):
            dtype_doc = self(v_type)
            name_doc = self(v)
            return dtype_doc + ' ' + name_doc
        elif isinstance(v_type, TensorType):
            if v_type.scope.name == 'shared':
                scope_doc = '__shared__ '
            else:
                scope_doc = ''
            dtype_doc = self(v_type.scalar_type)
            name_doc = self(v)
            shape_doc = Doc()
            for s in v_type.shape:
                shape_doc += '[' + self(s) + ']'
            return scope_doc + dtype_doc + ' ' + name_doc + shape_doc
        elif isinstance(v_type, PointerType):
            base_type_doc = self(v_type.base_type)
            name_doc = self(v)
            return base_type_doc + ' *' + name_doc
        elif isinstance(v_type, TensorPointerType):
            dtype_doc = self(v_type.tensor_type.scalar_type)
            name_doc = self(v)
            return dtype_doc + ' *' + name_doc
        else:
            assert False

    def __call__(self, node) -> Doc:
        return self.visit(node)

    def visit(self, node):
        if isinstance(node, IRModule):
            return self.visit_IRModule(node)
        elif isinstance(node, Function):
            return self.visit_Function(node)
        elif isinstance(node, (Stmt, Expr)):
            return StmtExprFunctor.visit(self, node)
        elif isinstance(node, TypeNode):
            return TypeFunctor.visit(self, node)
        elif isinstance(node, (tuple, list)):
            return doc_join([self(v) for v in node], ', ')
        elif isinstance(node, (int, float, bool)):
            return self(convert(node))
        else:
            raise ValueError(type(node))

    def visit_IRModule(self, module: IRModule) -> Doc:
        self.ir_module = module
        doc = Doc()
        doc += Text('#include <cassert>') + NewLine()
        doc += Text('#include <cstdio>') + NewLine()
        doc += Text('#include <cstdint>') + NewLine()
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

        # launch bound for grid worker
        if isinstance(worker, Grid):
            block_dim = simplify(worker.block_dim)
            if worker.min_blocks:
                min_blocks = simplify(worker.min_blocks)
            else:
                DEFAULT_MIN_BLOCKS = 2  # todo let user specify
                min_blocks = DEFAULT_MIN_BLOCKS
            if isinstance(block_dim, Constant):
                doc += f' __launch_bounds__({block_dim.value}, {min_blocks})'

        # func name
        canonized_func_name = self.canonize_funcname(func.name)
        doc += ' ' + canonized_func_name
        self.func_name_map[func.name] = canonized_func_name

        # parameters
        doc += '('
        param_docs = []
        for i in range(len(func.params)):
            param = func.params[i]
            param_docs.append(self.param_declare(param))
        doc += doc_join(param_docs, Text(', '))
        doc += ') {'

        # comments
        label = func.get_attr('label')
        if label:
            doc += (NewLine() + '// label: {}'.format(label)).indent()

        # locals
        for local_var in func.local_vars:
            doc += (NewLine() + self.local_var_declare(local_var) + ';').indent()
            # doc += (NewLine() + self(local_var.type) + ' ' + self(local_var) + ';').indent()

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

    def visit_TensorElement(self, e: TensorElement):
        return self(e.base) + doc_join(['[' + self(idx) + ']' for idx in e.indices], '')

    def visit_IfThenElse(self, e: IfThenElse):
        return '(' + self(e.cond) + ' ? ' + self(e.then_expr) + ' : ' + self(e.else_expr) + ')'

    def visit_Cast(self, e: Cast):
        return Text('(') + self.visit(e.target_type) + ')' + self(e.expr)

    def visit_Address(self, e: Address):
        return Text('&') + self.visit(e.expr)

    def visit_Reference(self, e: Reference):
        raise NotImplementedError()

    def visit_Dereference(self, e: Dereference):
        return Text('*') + self(e.expr)

    def visit_Call(self, e: Call):
        func_name = e.func_var.hint
        func_name = func_name.replace('.', '_')
        if func_name in self.ir_module.functions:
            # first check whether callee is in current ir module
            # because ir module functions will cover primitive functions
            func = self.ir_module.lookup(func_name)
        else:
            assert is_primitive_function(func_name), "Callee {} not found in current ir module, and it is not primitive function."
            v, func_type, func = get_primitive_function(func_name)
            assert func is None, "Please use import_primitive_functions pass to import primitive function first"
            # system-provided function, do not canonize the func name
            return func_name + (Text('(') + doc_join([self(arg) for arg in e.args], Text(', ')) + ')')
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

    def visit_Let(self, e: Let):
        raise ValueError("please run 'expand_let_expr' pass before codegen")

    def visit_Var(self, e: Var):
        cast2int = {
            'threadIdx.x',
            'blockIdx.x'
        }
        name = self.namer.get_name(e)
        if name in cast2int:
            return Text(f'(int){name}')
        else:
            return Text(name)

    def visit_Constant(self, e: Constant):
        if e.dtype.name == 'bool':
            return Text(str(e.value).lower())
        else:
            return Text(str(e.value))

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        return NewLine() + self(stmt.expr) + ';'

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        doc = NewLine()
        doc += self(stmt.buf)
        for idx in stmt.indices:
            doc += '[' + self(idx) + ']'
        doc += Text(' = ') + self(stmt.value) + ';'
        return doc

    def visit_AssignStmt(self, stmt: AssignStmt):
        return NewLine() + self(stmt.var) + ' = ' + self(stmt.value) + ';'

    def visit_LetStmt(self, stmt: LetStmt):
        doc = Doc()
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            doc += NewLine() + self(bind_var.type) + ' ' +  self(bind_var) + ' = ' + self(bind_value) + ';'
        doc += self(stmt.body)
        return doc

    def visit_ForStmt(self, stmt: ForStmt):
        v = stmt.loop_var
        init_doc = self(v.type) + ' ' + self(v) + ' = ' + self(convert(0))
        cond_doc = self(v < stmt.extent)
        update_doc = self(v) + ' = ' + self(v + 1)
        doc = Text('')
        if stmt.unroll is not None:
            if stmt.unroll:
                doc += NewLine() + '#pragma unroll'  # complete unroll
            else:
                doc += NewLine() + '#pragma unroll 1'  # prevent from unrolling
        doc += NewLine() + Text('for (') + init_doc + '; ' + cond_doc + '; ' + update_doc + ') '
        body_doc = self(stmt.body)
        doc += Text('{') + body_doc.indent() + NewLine() + Text('} ')
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

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        return NewLine() + 'return;'

    def visit_AssertStmt(self, stmt: AssertStmt):
        return NewLine() + Text('assert(((void)"') + stmt.msg + '", ' + self(stmt.cond) + '));'

    def visit_AsmStmt(self, stmt: AsmStmt):
        volatile_doc = 'volatile ' if stmt.is_volatile else ''
        template_doc = f'"{Text(stmt.template_string)}"'
        output_docs = []
        for label, expr in zip(stmt.output_labels, stmt.output_exprs):
            output_docs.append(Text(f'"{label}"') + '(' + self(expr) + ')')
        input_docs = []
        for label, expr in zip(stmt.input_labels, stmt.input_exprs):
            input_docs.append(Text(f'"{label}"') + '(' + self(expr) + ')')
        return NewLine() + 'asm ' + volatile_doc + '(' + template_doc + ' : ' + doc_join(output_docs, ', ') + ' : ' + doc_join(input_docs, ', ') + ');'

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        expr_docs = [str(self(e)) for e in stmt.exprs]
        stmt_string: str = stmt.template_string.format(*expr_docs)
        lines = stmt_string.split('\n')
        doc = Text('')
        for line in lines:
            doc += NewLine() + line
        return doc

    def visit_SeqStmt(self, stmt: SeqStmt):
        doc = Doc()
        for idx, s in enumerate(stmt.seq):
            doc += self(s)
        return doc

    def visit_ScalarType(self, t: ScalarType):
        scalar_type_map = {
            'int32': 'int32_t',
            'float32': 'float',
            'uint8': 'uint8_t'
        }
        return Text(scalar_type_map[t.name])

    def visit_TensorType(self, t: TensorType):
        return Text('TensorType(') + self(t.scalar_type) + ', [' + doc_join([self(s) for s in t.shape], ", ") + '], ' + t.scope.name + ')'

    def visit_PointerType(self, t: PointerType):
        return self(t.base_type) + Text('*')

    def visit_TensorPointerType(self, t: TensorPointerType):
        raise ValueError()

    def visit_ReferenceType(self, t: ReferenceType):
        raise ValueError()

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


def codegen(ir_module: IRModule) -> str:
    gen = Codegen()
    doc = gen(ir_module)
    return str(doc)
