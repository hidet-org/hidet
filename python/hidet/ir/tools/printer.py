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
from typing import Optional, List, Union, Dict, Tuple
from hidet.ir.node import Node
from hidet.ir.func import IRModule, Function
from hidet.ir.type import DataType, TensorType, VoidType, PointerType, ReferenceType, TensorPointerType
from hidet.ir.expr import Constant, Var, Call, TensorElement, Add, Multiply, LessThan, FloorDiv, Mod, Equal, Div
from hidet.ir.expr import Sub, LogicalNot, LogicalOr, LogicalAnd, Let, IfThenElse, TensorSlice
from hidet.ir.expr import RightShift, LeftShift, BitwiseNot, BitwiseOr
from hidet.ir.expr import BitwiseAnd, Neg, Cast, NotEqual, BitwiseXor, Reference, Dereference, Address
from hidet.ir.stmt import SeqStmt, IfStmt, ForStmt, AssignStmt, BufferStoreStmt, EvaluateStmt, AssertStmt
from hidet.ir.stmt import (
    BlackBoxStmt,
    AsmStmt,
    ReturnStmt,
    LetStmt,
    DeclareStmt,
    ForMappingStmt,
    WhileStmt,
    ContinueStmt,
)
from hidet.ir.stmt import BreakStmt, DeclareScope, LaunchKernelStmt
from hidet.ir.layout import StridesLayout, ConcatLayout, LocalLayout, SwizzleLayout, ComposedLayout, RowMajorLayout
from hidet.ir.layout import ColumnMajorLayout
from hidet.ir.mapping import RepeatTaskMapping, SpatialTaskMapping, ComposedTaskMapping
from hidet.ir.compute import TensorNode, GridCompute, ArgReduceCompute, ReduceCompute, TensorInput, ScalarInput
from hidet.ir.dialects.pattern import PlaceholderExpr
from hidet.ir.task import Task
from hidet.utils import same_list
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.utils.namer import Namer

from hidet.ir.functors import IRFunctor

_show_var_id = False


class IRPrinter(IRFunctor):
    def __init__(self):
        super().__init__(use_memo=False)
        self.namer = Namer()
        self.ir_module: Optional[IRModule] = None

    def __call__(self, node):
        return self.visit(node)

    def visit_Tuple(self, tp: Tuple):
        return doc_join([self(v) for v in tp], ', ')

    def visit_List(self, lst: List):
        return doc_join([self(v) for v in lst], ', ')

    def visit_Dict(self, d: Dict):
        return doc_join([k + ': ' + self(v) for k, v in d.items()], ', ')

    def visit_NotDispatchedNode(self, n: Node):
        raise NotImplementedError('Do not support print node {}'.format(type(n)))

    def visit_PyConstant(self, c: Union[str, int, float, None]):
        return Text(str(c))

    def visit_Function(self, func: Function):
        self.namer.clear()

        # parameters
        head_doc = Doc()
        head_doc += Text('def ') + func.name + '('
        for i, param in enumerate(func.params):
            head_doc += (NewLine() + self(param) + ': ' + self(param.type)).indent(4)
            if i < len(func.params) - 1:
                head_doc += ','
        head_doc += NewLine() + ')'

        # attributes
        attr_doc = Doc()
        for attr_name, attr_value in func.attrs.items():
            attr_doc += (NewLine() + '# {}: {}'.format(attr_name, attr_value)).indent(4)

        # body
        body_doc = self(func.body).indent(4)

        return head_doc + attr_doc + body_doc + NewLine()

    def visit_IRModule(self, ir_module: IRModule):
        doc = Doc()
        self.ir_module = ir_module
        if ir_module.task is not None:
            doc += self(ir_module.task)
        doc += NewLine()
        for func in ir_module.functions.values():
            doc += self(func) + NewLine()
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
        return '(~' + self(e.a) + ')'

    def visit_LeftShift(self, e: LeftShift):
        return '(' + self(e.a) + ' << ' + self(e.b) + ')'

    def visit_RightShift(self, e: RightShift):
        return '(' + self(e.a) + ' >> ' + self(e.b) + ')'

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
        if _show_var_id:
            return Text('{}@{}'.format(self.namer.get_name(e), e.id))
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
            doc += NewLine() + 'let ' + self(bind_var) + ': ' + self(bind_var.type) + ' = ' + self(bind_value)
        doc += self(stmt.body)
        # doc += self(stmt.body).indent()
        return doc

    def visit_ForStmt(self, stmt: ForStmt):
        rng = Text('range(') + self(stmt.extent) + ')'
        doc = NewLine() + Text('for ') + self(stmt.loop_var) + ' in ' + rng
        if stmt.attr.unroll is not None:
            doc += '  # ' + str(stmt.attr)
        doc += self(stmt.body).indent(4)
        return doc

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
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
        if stmt.ret_value is not None:
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
        return NewLine() + Text("{}<<<dim3({}, {}, {}), dim3({}, {}, {}), {}>>>({});").format(
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

    def _tensor_type(self, t: TensorType):
        items = [self(t.dtype), '[' + self(t.shape) + ']']
        if isinstance(t.layout, RowMajorLayout) or t.layout is None:
            # default layout, do not print
            pass
        else:
            items.append(self(t.layout))
        return doc_join(items, ', ')

    def visit_TensorType(self, t: TensorType):
        return Text('tensor(') + self._tensor_type(t) + ')'

    def visit_PointerType(self, t: PointerType):
        return Text('PointerType(') + self(t.base_type) + ')'

    def visit_TensorPointerType(self, t: TensorPointerType):
        return Text('tensor_pointer(') + self._tensor_type(t.tensor_type) + ')'

    def visit_ReferenceType(self, t: ReferenceType):
        return Text('ReferenceType(') + self(t.base_type) + ')'

    def visit_VoidType(self, t: VoidType):
        return Text('VoidType')

    def visit_AnyExpr(self, e: PlaceholderExpr):
        return Text('AnyExpr')

    def print_tensor_nodes(self, nodes: List[TensorNode], exclude_nodes: List[TensorNode] = None) -> Doc:
        from hidet.ir.tools import collect  # pylint: disable=import-outside-toplevel

        if exclude_nodes is None:
            exclude_nodes = []
        nodes: List[TensorNode] = collect(nodes, TensorNode)
        doc = Doc()
        for node in reversed(nodes):
            if node in exclude_nodes:
                continue
            if isinstance(node, TensorInput):
                doc += NewLine() + self(node) + ': ' + self(node.ttype)
            elif isinstance(node, GridCompute):
                # example
                # y: float32[10, 10] where y[i, j] = x[i, j] + 1
                doc += NewLine()
                doc += self(node) + ': ' + self(node.type.dtype) + '[' + self(node.type.shape) + ']'
                doc += Text(' where ') + self(node) + '[' + self(node.axes) + '] = ' + self(node.value)
            else:
                raise NotImplementedError()
        return doc

    def visit_Task(self, e: Task):
        lines = [
            Text('name: ') + e.name,
            Text('parameters: ')
            + (
                NewLine()
                + doc_join(['{}: {}'.format(self.namer.get_name(v), self(v.type)) for v in e.params], NewLine())
            ).indent(),
            Text('inputs: ') + '[' + doc_join([self.namer.get_name(v) for v in e.inputs], ', ') + ']',
            Text('outputs: ') + '[' + doc_join([self.namer.get_name(v) for v in e.outputs], ', ') + ']',
            Text('computations: ') + self.print_tensor_nodes(e.outputs).indent(),
            Text('attributes: {') + self({k: str(v) for k, v in e.attrs.items()}) + '}',
        ]
        front_part = doc_join(lines, NewLine())
        inverse_map_doc = Doc()
        if e.inverse_map:
            inverse_map_doc += NewLine() + Text('inverse_map:')
            for tensor, inverse_map in e.inverse_map.items():
                inverse_map_body = 'InverseMap([' + self(inverse_map.axes) + '] => [' + self(inverse_map.indices) + '])'
                inverse_map_doc += (NewLine() + self.namer.get_name(tensor) + ': ' + inverse_map_body).indent()
        return Text('Task(') + (NewLine() + front_part + inverse_map_doc).indent() + NewLine() + ')'

    def visit_TensorNode(self, e: TensorNode):
        return self.namer.get_name(e)

    def visit_ScalarInput(self, node: ScalarInput):
        return self.namer.get_name(node)

    def visit_TensorInput(self, node: TensorInput):
        return self.namer.get_name(node)

    def visit_GridCompute(self, c: GridCompute):
        return self.namer.get_name(c)

    def visit_ReduceCompute(self, c: ReduceCompute):
        items = ['[' + self(c.shape) + ']', '(' + self(c.axes) + ') => ' + self(c.value), str(c.reduce_operation)]
        return 'reduce(' + doc_join(items, ', ') + ')'

    def visit_ArgReduceCompute(self, c: ArgReduceCompute):
        items = ['[' + self(c.extent) + ']', self(c.axis) + ' => ' + self(c.value), str(c.reduce_operation)]
        return 'arg_reduce(' + doc_join(items, ', ') + ')'

    def visit_SpatialTaskMapping(self, mapping: SpatialTaskMapping):
        items = [self(mapping.task_shape)]
        if not same_list(mapping.ranks, list(range(len(mapping.task_shape)))):
            items.append('ranks=[' + self(mapping.ranks) + ']')
        return 'spatial(' + doc_join(items, ', ') + ')'

    def visit_RepeatTaskMapping(self, mapping: RepeatTaskMapping):
        items = [self(mapping.task_shape)]
        if not same_list(mapping.ranks, list(range(len(mapping.task_shape)))):
            items.append('ranks=[' + self(mapping.ranks) + ']')
        return 'repeat(' + doc_join(items, ', ') + ')'

    def visit_ComposedTaskMapping(self, mapping: ComposedTaskMapping):
        return self(mapping.outer) + '.' + self(mapping.inner)

    def visit_StridesLayout(self, layout: StridesLayout):
        if isinstance(layout, RowMajorLayout):
            return Text('row(') + self(layout.shape) + ')'
        elif isinstance(layout, ColumnMajorLayout):
            return Text('column(') + self(layout.shape) + ')'
        else:
            return Text('strides(') + self(layout.strides) + ')'

    def visit_SwizzleLayout(self, layout: SwizzleLayout):
        items = [self(layout.base), Text('dim=') + self(layout.dim), Text('regards=') + self(layout.regards_dim)]
        if layout.log_step != 0:
            items.append(Text('log_step=') + self(layout.log_step))
        return Text('swizzle(') + doc_join(items, ', ') + ')'

    def visit_LocalLayout(self, layout: LocalLayout):
        return Text('local(') + self(layout.shape) + ')'

    def visit_ComposedLayout(self, layout: ComposedLayout):
        return self(layout.outer) + ' * ' + self(layout.inner)

    def visit_ConcatLayout(self, layout: ConcatLayout):
        return Text('concat(') + self(layout.lhs) + ', ' + self(layout.rhs) + ')'


def astext(obj: Node) -> str:
    if isinstance(obj, Node):
        printer = IRPrinter()
        return str(printer(obj))
    else:
        raise ValueError()
