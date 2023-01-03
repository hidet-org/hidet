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
from typing import Optional, List
from hidet.ir.node import Node
from hidet.ir.func import IRModule, Function
from hidet.ir.type import DataType, TensorType, TypeNode, VoidType, PointerType, ReferenceType, TensorPointerType
from hidet.ir.expr import Constant, Var, Call, TensorElement, Add, Multiply, Expr, LessThan, FloorDiv, Mod, Equal, Div
from hidet.ir.expr import Sub, LogicalNot, LogicalOr, LogicalAnd, Let, IfThenElse, TensorSlice
from hidet.ir.expr import RightShift, LeftShift, BitwiseNot, BitwiseOr
from hidet.ir.expr import BitwiseAnd, Neg, Cast, NotEqual, BitwiseXor, Reference, Dereference, Address
from hidet.ir.stmt import (
    SeqStmt,
    IfStmt,
    ForStmt,
    AssignStmt,
    BufferStoreStmt,
    EvaluateStmt,
    Stmt,
    AssertStmt,
    LaunchKernelStmt,
)
from hidet.ir.stmt import BlackBoxStmt, AsmStmt, ReturnStmt, LetStmt, DeclareStmt, ForTaskStmt, WhileStmt, ContinueStmt
from hidet.ir.stmt import BreakStmt, DeclareScope
from hidet.ir.mapping import RepeatTaskMapping, SpatialTaskMapping, ComposedTaskMapping, TaskMapping
from hidet.ir.compute import TensorNode, ScalarNode, GridCompute, ArgReduceCompute, ReduceCompute
from hidet.ir.dialects.pattern import AnyExpr
from hidet.ir.layout import RowMajorLayout, ColumnMajorLayout
from hidet.ir.task import Task, TaskGraph, InverseMap
from hidet.utils import same_list
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.utils.namer import Namer

from .base import StmtExprFunctor, TypeFunctor, NodeFunctor


class IRPrinter(StmtExprFunctor, TypeFunctor):
    def __init__(self):
        super().__init__()
        self.namer = Namer()
        self.ir_module: Optional[IRModule] = None

    def __call__(self, node):
        return self.visit(node)

    def visit(self, obj):  # pylint: disable=arguments-renamed, too-many-branches
        # python builtin type
        if isinstance(obj, (list, tuple)):
            return doc_join([self(v) for v in obj], ', ')
        elif isinstance(obj, dict):
            return doc_join([self(k) + ': ' + self(v) for k, v in obj.items()], ', ')
        elif isinstance(obj, str):
            return Text(obj.replace('\n', '\\n').replace('\t', '\\t'))
        elif isinstance(obj, (int, float)):
            return Text(str(obj))
        elif obj is None:
            return Text('None')
        # type node
        elif isinstance(obj, TypeNode):
            return TypeFunctor.visit(self, obj)
        # function and ir module
        elif isinstance(obj, Function):
            return self.visit_Function(obj)
        elif isinstance(obj, IRModule):
            return self.visit_IRModule(obj)
        # expression and statement
        elif isinstance(obj, (Expr, Stmt)):
            return NodeFunctor.visit(self, obj)
        # task related
        elif isinstance(obj, Task):
            return self.visit_Task(obj)
        elif isinstance(obj, TaskGraph):
            return self.visit_TaskGraph(obj)
        elif isinstance(obj, InverseMap):
            return self.visit_InverseMap(obj)
        # task mapping
        elif isinstance(obj, TaskMapping):
            return self.visit_TaskMapping(obj)
        else:
            raise ValueError('Do not support print object {}'.format(object.__repr__(obj)))

    def visit_Function(self, func: Function):
        self.namer.clear()
        doc = Doc()

        # parameters
        doc += 'fn('
        param_docs = []
        for i, param in enumerate(func.params):
            line = []
            if i != 0:
                line.append(NewLine())
            line.extend([self(param), ': ', self(param.type)])
            param_docs.append(line)
        doc += doc_join(param_docs, Text(', '))
        doc += ')'
        doc = doc.indent(3)

        # attributes
        for attr_name, attr_value in func.attrs.items():
            doc += (NewLine() + '# {}: {}'.format(attr_name, attr_value)).indent(4)

        # body
        doc += self(func.body).indent(4)

        return doc

    def visit_IRModule(self, ir_module: IRModule):
        doc = Doc()
        self.ir_module = ir_module
        if ir_module.task is not None:
            doc += str(ir_module.task)
        doc += NewLine()
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

    def visit_Neg(self, e: Neg):
        return Text('(-') + self(e.a) + ')'

    def visit_LessThan(self, e: LessThan):
        return Text('(') + self(e.a) + ' < ' + self(e.b) + ')'

    def visit_LessEqual(self, e: LessThan):
        return Text('(') + self(e.a) + ' <= ' + self(e.b) + ')'

    def visit_NotEqual(self, e: NotEqual):
        return Text('(') + self(e.a) + ' != ' + self(e.b) + ')'

    def visit_Equal(self, e: Equal):
        return Text('(') + self(e.a) + ' == ' + self(e.b) + ')'

    def visit_And(self, e: LogicalAnd):
        return Text('(') + self(e.a) + ' && ' + self(e.b) + ')'

    def visit_Or(self, e: LogicalOr):
        return Text('(') + self(e.a) + ' || ' + self(e.b) + ')'

    def visit_Not(self, e: LogicalNot):
        return Text('!') + self(e.a)

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        return '(' + self(e.a) + ' & ' + self(e.b) + ')'

    def visit_BitwiseOr(self, e: BitwiseOr):
        return '(' + self(e.a) + ' | ' + self(e.b) + ')'

    def visit_BitwiseXor(self, e: BitwiseXor):
        return '(' + self(e.a) + ' ^ ' + self(e.b) + ')'

    def visit_BitwiseNot(self, e: BitwiseNot):
        return '(~' + self(e.base) + ')'

    def visit_LeftShift(self, e: LeftShift):
        return '(' + self(e.base) + ' << ' + self(e.cnt) + ')'

    def visit_RightShift(self, e: RightShift):
        return '(' + self(e.base) + ' >> ' + self(e.cnt) + ')'

    def visit_TensorElement(self, e: TensorElement):
        if e.protected:
            doc = self(e.base) + '.protect_read([' + self(e.indices) + '])'
        else:
            doc = self(e.base) + '[' + self(e.indices) + ']'
        return doc

    def visit_TensorSlice(self, e: TensorSlice):
        subscriptions = []
        for index, start, end in zip(e.indices, e.starts, e.ends):
            if index is not None:
                subscriptions.append(self(index))
            else:
                doc = Doc()
                if start is not None:
                    doc += self(start)
                doc += ':'
                if end is not None:
                    doc += self(end)
                subscriptions.append(doc)
        return self(e.base) + '[' + doc_join(subscriptions, ', ') + ']'

    def visit_IfThenElse(self, e: IfThenElse):
        return '(' + self(e.cond) + ' ? ' + self(e.then_expr) + ' : ' + self(e.else_expr) + ')'

    def visit_Call(self, e: Call):
        doc = Doc()
        # name
        doc += e.func_var.hint
        # launch
        func_name = e.func_var.hint
        if self.ir_module and func_name in self.ir_module.functions:
            func = self.ir_module.functions[func_name]
            if func.kind == 'cuda_kernel':
                doc += '<<<' + self(func.attrs['cuda_grid_dim']) + ', ' + self(func.attrs['cuda_block_dim']) + '>>>'
        # params
        doc += '(' + self(e.args) + ')'
        return doc

    def visit_Let(self, e: Let):
        return Text('let(') + self(e.var) + '=' + self(e.value) + ': ' + self(e.body) + ')'

    def visit_Cast(self, e: Cast):
        return Text('cast(') + self(e.target_type) + ', ' + self(e.expr) + ')'

    def visit_Reference(self, e: Reference):
        return Text('Ref(') + self(e.expr) + ')'

    def visit_Dereference(self, e: Dereference):
        return Text('*') + self(e.expr)

    def visit_Address(self, e: Address):
        return Text('&') + self(e.expr)

    def visit_Var(self, e: Var):
        return Text(self.namer.get_name(e))

    def visit_Constant(self, e: Constant):
        if e.value is None:
            return self('Constant(None, type=') + self(e.type) + ')'
        if e.is_tensor():
            return 'ConstTensor({}, {})'.format(e.value.shape, e.type)
        else:
            dtype = e.type.name
            if dtype == 'float32':
                ret = '{}f'.format(float(e.value))
            elif dtype == 'float16':
                ret = 'half({})'.format(float(e.value))
            elif dtype == 'int32':
                ret = '{}'.format(int(e.value))
            else:
                ret = '{}({})'.format(dtype, e.value)
            return Text(ret)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        doc = NewLine() + Text('declare ') + self(stmt.var) + Text(': ') + self(stmt.var.type)
        if stmt.init is not None:
            doc += ' = ' + self(stmt.init)
        if stmt.is_static:
            doc += ' [static]'
        if stmt.scope != DeclareScope.Default:
            doc += ' [{}]'.format(stmt.scope)
        return doc

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        return NewLine() + self(stmt.expr)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        doc = NewLine()
        doc += self(stmt.buf)
        doc += '[' + self(stmt.indices) + ']'
        doc += ' = ' + self(stmt.value)
        if stmt.protected:
            doc += '  [protected write]'
        return doc

    def visit_AssignStmt(self, stmt: AssignStmt):
        return NewLine() + self(stmt.var) + ' = ' + self(stmt.value)

    def visit_LetStmt(self, stmt: LetStmt):
        doc = Doc()
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            doc += NewLine() + 'let ' + self(bind_var) + ' = ' + self(bind_value)
        doc += self(stmt.body)
        # doc += self(stmt.body).indent()
        return doc

    def visit_ForStmt(self, stmt: ForStmt):
        rng = Text('range(') + self(stmt.extent) + ')'
        doc = NewLine() + Text('for ') + self(stmt.loop_var) + ' in ' + rng
        if stmt.unroll is not None:
            if stmt.unroll:
                doc += '[unroll]'
            else:
                doc += '[no-unroll]'
        doc += self(stmt.body).indent(4)
        return doc

    def visit_ForTaskStmt(self, stmt: ForTaskStmt):
        doc = NewLine() + Text('for ') + self(stmt.loop_vars) + ' in ' + self(stmt.mapping) + ' on ' + self(stmt.worker)
        doc += self(stmt.body).indent(4)
        return doc

    def visit_WhileStmt(self, stmt: WhileStmt):
        doc = NewLine() + 'while ' + self(stmt.cond)
        doc += self(stmt.body).indent(4)
        return doc

    def visit_BreakStmt(self, stmt: BreakStmt):
        return NewLine() + 'break'

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        return NewLine() + 'continue'

    def visit_IfStmt(self, stmt: IfStmt):
        doc = NewLine() + Text('if ') + self(stmt.cond)
        doc += self(stmt.then_body).indent(4)
        if stmt.else_body:
            doc += NewLine() + Text('else')
            doc += self(stmt.else_body).indent(4)
        return doc

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        doc = NewLine() + Text('return')
        if stmt.ret_value:
            doc += ' ' + self(stmt.ret_value)
        return doc

    def visit_AssertStmt(self, stmt: AssertStmt):
        if stmt.msg:
            return NewLine() + 'assert(' + self(stmt.cond) + ', ' + stmt.msg + ')'
        else:
            return NewLine() + 'assert(' + self(stmt.cond) + ')'

    def visit_AsmStmt(self, stmt: AsmStmt):
        volatile_doc = 'volatile ' if stmt.is_volatile else ''
        template_doc = '"' + Text(stmt.template_string) + '"'
        output_docs = []
        for label, expr in zip(stmt.output_labels, stmt.output_exprs):
            output_docs.append('"' + Text(label) + '"' + '(' + self(expr) + ')')
        input_docs = []
        for label, expr in zip(stmt.input_labels, stmt.input_exprs):
            input_docs.append('"' + Text(label) + '"' + '(' + self(expr) + ')')
        return (
            NewLine()
            + 'asm '
            + volatile_doc
            + '('
            + template_doc
            + ' : '
            + doc_join(output_docs, ', ')
            + ' : '
            + doc_join(input_docs, ', ')
            + ');'
        )

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        return Text("{}<<<dim3({}, {}, {}), dim3({}, {}, {}), {}>>>({});").format(
            self(stmt.func_var),
            self(stmt.grid_dim[0]),
            self(stmt.grid_dim[1]),
            self(stmt.grid_dim[2]),
            self(stmt.block_dim[0]),
            self(stmt.block_dim[1]),
            self(stmt.block_dim[2]),
            self(stmt.shared_mem_bytes),
            self(stmt.args),
        )

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
        for s in stmt.seq:
            doc += self(s)
        return doc

    def visit_ScalarType(self, t: DataType):
        return Text('{}'.format(t.name))

    def visit_TensorType(self, t: TensorType):
        items = [self(t.dtype), '[' + self(t.shape) + ']']
        if isinstance(t.layout, RowMajorLayout):
            # default layout, do not print
            pass
        elif isinstance(t.layout, ColumnMajorLayout):
            items.append(Text('col_major'))
        elif t.layout is None:
            # skip None
            pass
        else:
            items.append(Text(type(t.layout).__name__))
        return Text('tensor(') + doc_join(items, ', ') + ')'

    def visit_PointerType(self, t: PointerType):
        return Text('PointerType(') + self(t.base_type) + ')'

    def visit_TensorPointerType(self, t: TensorPointerType):
        return Text('TensorPointerType(') + self(t.tensor_type) + ')'

    def visit_ReferenceType(self, t: ReferenceType):
        return Text('ReferenceType(') + self(t.base_type) + ')'

    def visit_VoidType(self, t: VoidType):
        return Text('VoidType')

    def visit_AnyExpr(self, e: AnyExpr):
        return Text('AnyExpr')

    def print_tensor_nodes(self, nodes: List[TensorNode], exclude_nodes: List[TensorNode] = None) -> Doc:
        from hidet.ir.functors import collect  # pylint: disable=import-outside-toplevel

        if exclude_nodes is None:
            exclude_nodes = []
        nodes: List[TensorNode] = collect(nodes, TensorNode)
        doc = Doc()
        for node in reversed(nodes):
            if node in exclude_nodes:
                continue
            if node.tensor_compute is None:
                doc += NewLine() + self(node) + ': ' + self(node.ttype)
            else:
                if isinstance(node.tensor_compute, GridCompute):
                    # example
                    # y: float32[10, 10] where y[i, j] = x[i, j] + 1
                    gc = node.tensor_compute
                    doc += NewLine()
                    doc += self(node) + ': ' + self(node.ttype.dtype) + '[' + self(node.ttype.shape) + ']'
                    doc += Text(' where ') + self(node) + '[' + self(gc.axes) + '] = ' + self(gc.value)
                    # items = [
                    #     '[' + self(gc.shape) + ']',
                    #     'where ',
                    #     '[' + self(gc.axes) + '] = ' + self(gc.value),
                    # ]
                    # doc += NewLine() + self.namer.get_name(node) + ': ' + 'grid(' + doc_join(items, ', ') + ')'
                else:
                    raise NotImplementedError()
        return doc

    def visit_Task(self, e: Task):
        lines = [
            Text('name: ') + e.name,
            Text('parameters: ')
            + (
                NewLine()
                + doc_join(['{}: {}'.format(self.namer.get_name(v), self(v.ttype)) for v in e.parameters], NewLine())
            ).indent(),
            Text('inputs: ') + '[' + doc_join([self.namer.get_name(v) for v in e.inputs], ', ') + ']',
            Text('outputs: ') + '[' + doc_join([self.namer.get_name(v) for v in e.outputs], ', ') + ']',
            Text('computations: ') + self.print_tensor_nodes(e.outputs).indent(),
            Text('attributes: {') + self({k: str(v) for k, v in e.attributes.items()}) + '}',
        ]
        if len(e.task_graph.nodes) > 1:
            lines.append(Text('task_graph: ') + self(e.task_graph))
        front_part = doc_join(lines, NewLine())
        inverse_map_doc = Doc()
        if e.inverse_map:
            inverse_map_doc += NewLine() + Text('inverse_map:')
            for tensor, inverse_map in e.inverse_map.items():
                inverse_map_doc += (NewLine() + self.namer.get_name(tensor) + ': ' + self(inverse_map)).indent()
        return Text('Task(') + (NewLine() + front_part + inverse_map_doc).indent() + NewLine() + ')'

    def visit_TaskGraph(self, task_graph: TaskGraph):
        head = Text('TaskGraph(') + self(task_graph.input_tensors) + ') {'
        body = []
        for task in task_graph.nodes:
            arg_items = []
            for task_input in task.inputs:
                if task_input in task_graph.consume:
                    arg_items.append(self(task_input) + '=' + self(task_graph.consume[task_input]))
                else:
                    arg_items.append(self(task_input))
                # task_input = task_graph.consume[task_input] if task_input in task_graph.consume else task_input
                # arg_items.append(self(task_input) + '=' + )
            for name, value in task.attributes.items():
                arg_items.append(self(name) + '=' + self(str(value)))
            args = doc_join(arg_items, ', ')
            assign_line = self(task.outputs) + ' = ' + task.name + '(' + args + ')'
            if task is task_graph.anchor:
                assign_line = assign_line + ' [anchor]'
            if task is task_graph.anchor:
                compute_body = Doc()
            else:
                compute_body = self.print_tensor_nodes(task.outputs, exclude_nodes=task.inputs).indent()
            body.append(assign_line + compute_body)

        body.append(
            'return '
            + self([task_graph.consume[v] if v in task_graph.consume else v for v in task_graph.output_tensors])
        )

        body = (NewLine() + doc_join(body, NewLine())).indent()
        tail = NewLine() + '}'
        return head + body + tail

    def visit_InverseMap(self, e: InverseMap):
        return 'InverseMap([' + self(e.axes) + '] => [' + self(e.indices) + '])'

    def visit_ScalarNode(self, e: ScalarNode):
        if e.scalar_compute is None:
            return self.namer.get_name(e, e.name)
        else:
            sc = e.scalar_compute
            if isinstance(sc, ReduceCompute):
                items = [
                    '[' + self(sc.shape) + ']',
                    '(' + self(sc.axes) + ') => ' + self(sc.value),
                    str(sc.reduce_operation),
                ]
                return 'reduce(' + doc_join(items, ', ') + ')'
            elif isinstance(sc, ArgReduceCompute):
                items = [
                    '[' + self(sc.extent) + ']',
                    '' + self(sc.axis) + ' => ' + self(sc.value),
                    str(sc.reduce_operation),
                ]
                return 'arg_reduce(' + doc_join(items, ', ') + ')'
            else:
                raise NotImplementedError()

    def visit_TensorNode(self, e: TensorNode):
        return self.namer.get_name(e)

    def visit_TaskMapping(self, mapping: TaskMapping):
        if isinstance(mapping, (RepeatTaskMapping, SpatialTaskMapping)):
            name = 'repeat' if isinstance(mapping, RepeatTaskMapping) else 'spatial'
            args = [self(mapping.task_shape)]
            if not same_list(mapping.ranks, list(range(len(mapping.task_shape)))):
                args.append('ranks=[' + self(mapping.ranks) + ']')
            arg_doc = doc_join(args, ', ')
            # something like: spatial(1, 3, ranks=[1, 0])
            return doc_join([name, '(', arg_doc, ')'], '')
        elif isinstance(mapping, ComposedTaskMapping):
            return self(mapping.outer) + '.' + self(mapping.inner)
        else:
            raise NotImplementedError()


def astext(obj: Node) -> str:
    if isinstance(obj, Node):
        printer = IRPrinter()
        return str(printer(obj))
    else:
        raise ValueError()
