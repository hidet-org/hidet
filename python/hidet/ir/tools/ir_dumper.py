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
# %%
from typing import Optional, List, Union, Dict, Tuple, Any, Set

import hidet.utils.structure
from hidet import ir
from hidet.ir.node import Node
from hidet.ir.expr import Expr, SymbolVar
from hidet.ir.module import IRModule
from hidet.ir.func import Function
from hidet.ir.type import DataType, TensorType, VoidType, PointerType, ReferenceType, TensorPointerType, FuncType
from hidet.ir.type import ArrayType, StringType
from hidet.ir.expr import Constant, Var, Call, TensorElement, Add, Multiply, LessThan, FloorDiv, Mod, Equal, Div
from hidet.ir.expr import Sub, LogicalNot, LogicalOr, LogicalAnd, Let, IfThenElse, TensorSlice
from hidet.ir.expr import RightShift, LeftShift, BitwiseNot, BitwiseOr
from hidet.ir.expr import BitwiseAnd, Neg, Cast, NotEqual, BitwiseXor, Reference, Dereference, Address
from hidet.ir.stmt import SeqStmt, IfStmt, ForStmt, AssignStmt, BufferStoreStmt, EvaluateStmt, AssertStmt
from hidet.ir.stmt import BlackBoxStmt, AsmStmt, ReturnStmt, LetStmt, DeclareStmt, ForMappingStmt, WhileStmt
from hidet.ir.stmt import BreakStmt, DeclareScope, LaunchKernelStmt, ContinueStmt
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
from hidet.ir.primitives import is_primitive_function, lookup_primitive_function


class IRDumper(IRFunctor):
    def __init__(self):
        super().__init__(use_memo=False)
        self.namer = Namer()
        self.ir_module: Optional[IRModule] = None
        self.attr_table: Dict[str, Any] = {}
        self.module_headers: List[str] = []
        self.global_symbolvars: Set[SymbolVar] = set()

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
    
    def clear_attr_table(self):
        self.attr_table = {}
        self.module_headers = []
    
    def format_attr_dict(self, name: str, d: Dict) -> str:
        if name in self.attr_table:
            raise ValueError('Attribute dict with name {} already exists'.format(name))
        def check_attr(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if not isinstance(k, str):
                        raise ValueError('Attribute dict must have key of string, but got {}'.format(type(k)))
                    check_attr(v)
            elif isinstance(d, (list, tuple)):
                for it in d:
                    check_attr(it)
            elif isinstance(d, (int, float, str, bool, Constant, Expr)):
                pass
            elif d is None:
                pass
            else:
                raise ValueError("Unexpected type {} in attribute dict".format(type(d)))
        if not isinstance(d, dict):
            raise ValueError('Attribute dict must be a dict, but got {}'.format(type(d)))
        check_attr(d)

        self.attr_table[name] = d
        
        def format_str(d):
            if isinstance(d, dict):
                return 'dict(' + ', '.join([format_str(k) + ': ' + format_str(v) for k, v in d.items()]) + ')'
            elif isinstance(d, (list, tuple)):
                return 'list(' + ', '.join([format_str(it) for it in d]) + ')'
            elif isinstance(d, str):
                return '"{}"'.format(d)
            elif isinstance(d, Expr):
                return str(self(d))
            elif d is True:
                return 'true'
            elif d is False:
                return 'false'
            elif d is None:
                return 'none'
            else:
                return str(d)

        return f"#{name} = " + format_str(d) + ";"
    
    def format_attr_name(self, name: str) -> str:
        return f"#{name}"
    
    def get_unique_attr_name(self, name: str) -> str:
        if name not in self.attr_table:
            return name
        i = 1
        while name + str(i) in self.attr_table:
            i += 1
        return name + str(i)

    def add_attr(self, name: str, d: Dict) -> Tuple[str, Optional[str]]:
        """
        Each attribute name references to a unique attribute somewhere higher.
        If the name does not exist yet, the create a new attribute,
        if the name exists and the attributes are not equal, create a new attribute with a new name.
        If the name exists and the attributes are equal, just refer to the existing attribute.
        """
        def dict_equal(d1, d2):
            if len(d1) != len(d2):
                return False
            for k, v0 in d1.items():
                if k not in d2:
                    return False
                v1 = d2[k]
                if isinstance(v0, dict):
                    if not dict_equal(v0, v1):
                        return False
                elif isinstance(v0, (list, tuple)):
                    if not same_list(v0, v1):
                        return False
                elif isinstance(v0, Node) or isinstance(v1, Node):
                    return False
                elif v0 != v1:
                    return False
            return True
        if name not in self.attr_table:
            # do this to preserve the order of defined attributes
            self.module_headers.append(self.format_attr_dict(name, d))
            return self.format_attr_name(name)
        else:
            
            if self.attr_table[name] == d:
                return self.format_attr_name(name)
            i = 1
            while name + str(i) in self.attr_table:
                if self.attr_table[name + str(i)] == d:
                    return self.format_attr_name(name + str(i))
                i += 1
            new_name = self.get_unique_attr_name(name)
            self.module_headers.append(self.format_attr_dict(new_name, d))
            return self.format_attr_name(new_name)

    def visit_Function(self, func: Function):
        self.namer.clear()

        # parameters
        if len(func.attrs) > 0:
            fn_attr_name = self.add_attr('f', func.attrs)
        else:
            fn_attr_name = ''

        head_doc = NewLine()
        head_doc += Text('def ') + func.name + '('
        for i, param in enumerate(func.params):
            head_doc += (NewLine() + self(param) + ': ' + self(param.type)).indent(4)
            if i < len(func.params) - 1:
                head_doc += ','
            else:
                head_doc += NewLine()
        head_doc += ')'
        head_doc += ' -> ' + self(func.ret_type) + ' ' + fn_attr_name + ' {'

        # body
        body_doc = self(func.body).indent(4)

        return head_doc + body_doc + NewLine() + "}" + NewLine()

    def visit_IRModule(self, ir_module: IRModule):
        doc = Doc()
        self.ir_module = ir_module

        ir_module_attrs = {"link_lib": ir_module.linking_libs, "external_object": ir_module.object_files, "namespace": ir_module.namespace,
                           "include_header": ir_module.include_headers}
        mod_attr_name =  self.get_unique_attr_name('m')
        doc += Text("module #") + mod_attr_name + " {" + NewLine()

        mod_inner_attr = NewLine()
        mod_body = Doc()
        for name, var in ir_module.global_vars.items():
            if name in ir_module.functions:
                continue
            mod_body += Text('decl ') + self(var) + Text(': ') + self(var.type) + ";" + NewLine() + NewLine()
        for func in ir_module.functions.values():
            mod_body += self(func) + NewLine()
        
        for i in self.module_headers:
            mod_inner_attr += i + NewLine()

        doc += mod_inner_attr.indent(4) + mod_body.indent(4)
        doc += NewLine() + "}" + NewLine()

        ir_module_attrs['symbols'] = {str(v): str(self(v.type)) for v in self.global_symbolvars}

        self.clear_attr_table()
        self.add_attr(mod_attr_name, ir_module_attrs)
        headers = Doc()
        for i in self.module_headers:
            headers += i + NewLine()
        
        self.global_symbolvars.clear()
        res = headers + doc
        return res

    def visit_TensorElement(self, e: TensorElement):
        # TODO: 
        if e.protected:
            doc = NewLine() + "protected " + self(e.base) + '[' + self(e.indices) + ']'
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
        return " if " + self(e.cond) + " then {" + self(e.then_expr) + "} else {" + self(e.else_expr) + "}"

    def visit_Call(self, e: Call):
        doc = Doc()
        print(e.func_var.__module__)
        func_name = e.func_var.name if e.func_var.name else e.func_var.hint
        # name
        doc += func_name
        # launch
        if self.ir_module and func_name in self.ir_module.functions:
            func = self.ir_module.functions[func_name]
            if func.kind == 'cuda_kernel':
                doc += '<<<' + self(func.attrs['cuda.grid_dim']) + ', ' + self(func.attrs['cuda.block_dim']) + '>>>'
        # params
        doc += '(' + self(e.args) + ')'
        return doc

    def visit_Let(self, e: Let):
        return Text('let(') + self(e.var) + ": " + self(e.var.type) + '=' + self(e.value) + ') in (' + self(e.body) + ')'

    def visit_Cast(self, e: Cast):
        return Text('cast(') + self(e.expr) + " as " + self(e.target_type) + ')'

    # def visit_Reference(self, e: Reference):
    #     return Text('Ref(') + self(e.expr) + ')'

    def visit_Dereference(self, e: Dereference):
        return Text('deref(') + self(e.expr) + ')'

    def visit_Address(self, e: Address):
        return Text('addr(') + self(e.expr) + ')'

    def visit_Var(self, e: Var):
        # if self.show_var_id:
        #     return Text('{}@{}'.format(self.namer.get_name(e), e.id))
        # print(e.__module__, type(e))
        if isinstance(e, SymbolVar):
            self.global_symbolvars.add(e)
            return Text('@' + self.namer.get_name(e))
        return Text(self.namer.get_name(e))

    def visit_Constant(self, e: Constant):
        if e.value is None:
            raise ValueError('Constant value can not be None.')
            # return self('Constant(None, ') + self(e.type) + ')'
        if e.is_tensor():
            raise NotImplementedError("Constant tensor is not supported yet.")
            # return 'ConstTensor({}, {})'.format(e.value.shape, e.type)
        elif e.is_string():
            return Text('"{}"'.format(str(e.value)))
        elif e.is_scalar():
            dtype = e.type.name
            if dtype == 'float32':
                ret = '{}'.format(float(e.value))
            elif dtype == 'float16':
                ret = '{}'.format(float(e.value))
            elif dtype == 'int32':
                ret = '{}'.format(int(e.value))
            elif dtype == 'bool':
                ret = 'true' if e.value else 'false'
            else:
                # ret = '{}({})'.format(dtype, e.value)
                raise NotImplementedError("Unknown constant type: {}".format(e.type))
            return Text(ret)
        elif isinstance(e.type, PointerType):
            return Text('cast({} as {})'.format(self(e.value), self(e.type)))
        else:
            raise NotImplementedError("Unknown constant type: {}".format(e.type))

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        doc = NewLine()
        if stmt.is_static:
            doc += 'static '
        if stmt.scope != DeclareScope.Default:
            attr = ' ' + self.add_attr("d", {"scope": str(stmt.scope)})
        else:
            attr = ''
        doc += Text('decl ') + self(stmt.var) + Text(': ') + self(stmt.var.type)

        if stmt.init is not None:
            doc += ' = ' + self(stmt.init)
        return doc + ';' + attr

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        return NewLine() + self(stmt.expr) + ';'

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        doc = NewLine()
        if stmt.protected:
            doc += ' protected '
        doc += self(stmt.buf)
        doc += '[' + self(stmt.indices) + ']'
        doc += ' = ' + self(stmt.value) + ';'
        return doc

    def visit_AssignStmt(self, stmt: AssignStmt):
        return NewLine() + self(stmt.var) + ' = ' + self(stmt.value) + ';'

    def visit_LetStmt(self, stmt: LetStmt):
        doc = NewLine()
        doc += NewLine() + Text('let(')
        doc_stmt = Doc()
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            doc_stmt += NewLine() + self(bind_var) + ': ' + self(bind_var.type) + ' = ' + self(bind_value) + ';'
        doc += doc_stmt.indent(4)
        doc += NewLine() + ') in ('
        doc += self(stmt.body).indent(4)
        doc += NewLine() + ');'
        # doc += self(stmt.body).indent()
        return doc
            

    def visit_ForStmt(self, stmt: ForStmt):
        for_stmt_attr = {}
        if stmt.attr.unroll or stmt.attr.parallel:
            if stmt.attr.unroll:
                for_stmt_attr['unroll'] = True
                for_stmt_attr['extent'] = stmt.extent
                for_stmt_attr['factor'] = stmt.attr.unroll_factor
            elif stmt.attr.parallel:
                for_stmt_attr['parallel'] = True
                for_stmt_attr['threads'] = stmt.attr.parallel_threads
        
        if len(for_stmt_attr) > 0:
            attr_name = self.add_attr('u', for_stmt_attr)
        else:
            attr_name = ''
        
        rng = Text('range(') + self(stmt.extent) + ') '
        doc = NewLine()
        doc += NewLine() + Text('for (') + self(stmt.loop_var) + ') in ' + rng + attr_name + ' {'
        doc += self(stmt.body).indent(4)
        doc += NewLine() + '}'
        return doc

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        doc = NewLine() + NewLine() + Text('for (') + self(stmt.loop_vars) + ') in ' + self(stmt.mapping) + ' on (' + self(stmt.worker) + ') {'
        doc += self(stmt.body).indent(4)
        doc += NewLine() + '}'
        return doc

    def visit_WhileStmt(self, stmt: WhileStmt):
        doc = NewLine() + 'while ' + self(stmt.cond) + ' {'
        doc += self(stmt.body).indent(4)
        doc += NewLine() + '}'
        return doc

    def visit_BreakStmt(self, stmt: BreakStmt):
        return NewLine() + 'break;'

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        return NewLine() + 'continue;'

    def visit_IfStmt(self, stmt: IfStmt):
        doc = NewLine() + Text('if ') + self(stmt.cond) + ' {'
        doc += self(stmt.then_body).indent(4)
        doc += NewLine() + '}'

        if stmt.else_body:
            doc += Text(' else {')
            doc += self(stmt.else_body).indent(4)
            doc += NewLine() + '}'
        return doc

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        doc = NewLine() + Text('return')
        if stmt.ret_value is not None:
            doc += ' ' + self(stmt.ret_value)
        doc += ';'
        return doc

    def visit_AssertStmt(self, stmt: AssertStmt):
        if stmt.msg:
            return NewLine() + 'assert(' + self(stmt.cond) + ', ' + repr(stmt.msg) + ');'
        else:
            return NewLine() + 'assert(' + self(stmt.cond) + ');'

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
            + volatile_doc
            + 'asm '
            + '{'
            + template_doc
            + ' { '
            + doc_join(output_docs, ', ')
            + ' } { '
            + doc_join(input_docs, ', ')
            + '} };'
        )

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        return NewLine() + Text("{}<<<({}, {}, {}), ({}, {}, {}), {}>>>({});").format(
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
        expr_docs = [self(e) for e in stmt.exprs]
        doc = NewLine() + Text('blackbox (') + doc_join(expr_docs, ', ') + ') {'
        inner = NewLine() + Text('$') + Text(stmt.template_string) + '$'
        doc += inner.indent(4) + NewLine() + '};'
        return doc

    def visit_SeqStmt(self, stmt: SeqStmt):
        doc = Doc()
        for s in stmt.seq:
            doc += self(s)
        # return NewLine() + Text('{') + doc.indent(4) + NewLine() + '}'
        return doc

    def visit_DataType(self, t: DataType):
        return Text('{}'.format(t.short_name))

    def _tensor_type(self, t: TensorType):
        layout = ''
        if isinstance(t.layout, RowMajorLayout) or t.layout is None:
            # default layout, do not print
            pass
        else:
            layout += '; ' + self(t.layout)
        layout = self(t.dtype) + '<' + self(t.shape) + layout + '>'
        return layout

    def visit_TensorType(self, t: TensorType):
        return self._tensor_type(t)

    # def visit_ArrayType(self, t: ArrayType):
    #     return Text('array(') + self(t.base_type) + ', size=' + self(t.size) + ')'

    def visit_StringType(self, t: StringType):
        return Text('str')

    def visit_PointerType(self, t: PointerType):
        # if isinstance(t.base_type, VoidType):
        #     return Text('~void')
        # if isinstance(t.base_type, (DataType, PointerType)):
        #     return '~' +self(t.base_type)
        return Text('~') + self(t.base_type)

    def visit_TensorPointerType(self, t: TensorPointerType):
        return Text('~') + self._tensor_type(t.tensor_type)

    def visit_ReferenceType(self, t: ReferenceType):
        return Text('~') + self(t.base_type)

    def visit_VoidType(self, t: VoidType):
        return Text('void')

    # def visit_FuncType(self, t: FuncType):
    #     if t.type_infer_func is not None:
    #         return Text('FuncType[type_infer_func]')
    #     else:
    #         return Text('FuncType(params={}, ret={})'.format(self(t.param_types), self(t.ret_type)))

    # def visit_PlaceholderExpr(self, e: PlaceholderExpr):
    #     if e.required_type:
    #         type_doc = self(e.required_type) + '_'
    #     else:
    #         type_doc = ''

    #     if e.require_const:
    #         base = 'const'
    #     elif e.require_non_const:
    #         base = 'expr'
    #     else:
    #         base = 'any'

    #     return Text(type_doc + base)

    # def print_tensor_nodes(self, nodes: List[TensorNode], exclude_nodes: List[TensorNode] = None) -> Doc:
    #     from hidet.ir.tools import collect  # pylint: disable=import-outside-toplevel
    #     from hidet.utils.structure import DirectedGraph

    #     if exclude_nodes is None:
    #         exclude_nodes = []
    #     nodes: List[TensorNode] = collect(nodes, TensorNode)
    #     dag = DirectedGraph()
    #     for node in nodes:
    #         dag.add_node(node)
    #         if isinstance(node, GridCompute):
    #             depends = collect(node.value, TensorNode, stop_when_found=True)
    #         elif isinstance(node, TensorInput):
    #             depends = []
    #         else:
    #             raise NotImplementedError()
    #         for depend_node in depends:
    #             dag.add_edge(src=depend_node, dst=node)
    #     order = dag.topological_order()

    #     doc = Doc()
    #     for node in order:
    #         if node in exclude_nodes:
    #             continue
    #         if isinstance(node, TensorInput):
    #             pass
    #         elif isinstance(node, GridCompute):
    #             # example
    #             # y: float32[10, 10] where y[i, j] = x[i, j] + 1
    #             doc += NewLine()
    #             doc += self(node) + ': ' + self(node.type.dtype) + '[' + self(node.type.shape) + ']'
    #             doc += Text(' where ') + self(node) + '[' + self(node.axes) + '] = ' + self(node.value)
    #         else:
    #             raise NotImplementedError()
    #     return doc

    # def visit_Task(self, e: Task):
    #     lines = [
    #         Text('name: ') + e.name,
    #         Text('parameters: ')
    #         + (
    #             NewLine()
    #             + doc_join(['{}: {}'.format(self.namer.get_name(v), self(v.type)) for v in e.params], NewLine())
    #         ).indent(),
    #         Text('inputs: ') + '[' + doc_join([self.namer.get_name(v) for v in e.inputs], ', ') + ']',
    #         Text('outputs: ') + '[' + doc_join([self.namer.get_name(v) for v in e.outputs], ', ') + ']',
    #         Text('computations: ') + self.print_tensor_nodes(e.outputs).indent(),
    #         Text('attributes: {') + self({k: str(v) for k, v in e.attrs.items()}) + '}',
    #     ]
    #     if len(e.assertions) > 0:  # between computations and attributes
    #         lines.append(
    #             Text('assertions: ')
    #             + (
    #                 NewLine()  # self.assertions: List[Tuple[Expr, str]]
    #                 + doc_join(['assert {}'.format(str(self.visit(v[0]))) for v in e.assertions], NewLine())
    #             ).indent()
    #         )
    #     front_part = doc_join(lines, NewLine())
    #     inverse_map_doc = Doc()
    #     if e.inverse_map:
    #         inverse_map_doc += NewLine() + Text('inverse_map:')
    #         for tensor, inverse_map in e.inverse_map.items():
    #             inverse_map_body = 'InverseMap([' + self(inverse_map.axes) + '] => [' + self(inverse_map.indices) + '])'
    #             inverse_map_doc += (NewLine() + self.namer.get_name(tensor) + ': ' + inverse_map_body).indent()
    #     return Text('Task(') + (NewLine() + front_part + inverse_map_doc).indent() + NewLine() + ')'

    # def visit_TensorNode(self, e: TensorNode):
    #     return self.namer.get_name(e)

    # def visit_ScalarInput(self, node: ScalarInput):
    #     return self.namer.get_name(node)

    # def visit_TensorInput(self, node: TensorInput):
    #     return self.namer.get_name(node)

    # def visit_GridCompute(self, c: GridCompute):
    #     return self.namer.get_name(c)

    # def visit_ReduceCompute(self, c: ReduceCompute):
    #     items = ['[' + self(c.shape) + ']', '(' + self(c.axes) + ') => ' + self(c.value), str(c.reduce_operation)]
    #     return 'reduce(' + doc_join(items, ', ') + ')'

    # def visit_ArgReduceCompute(self, c: ArgReduceCompute):
    #     items = ['[' + self(c.extent) + ']', self(c.axis) + ' => ' + self(c.value), str(c.reduce_operation)]
    #     return 'arg_reduce(' + doc_join(items, ', ') + ')'

    def visit_SpatialTaskMapping(self, mapping: SpatialTaskMapping):
        items = [self(mapping.task_shape)]
        if not same_list(mapping.ranks, list(range(len(mapping.task_shape)))):
            items.append('ranks=[' + self(mapping.ranks) + ']')
        attr = self.add_attr("c", {"fn_type": "TaskMapping"})
        return f'spatial_map{attr}(' + doc_join(items, ', ') + ')'

    def visit_RepeatTaskMapping(self, mapping: RepeatTaskMapping):
        items = [self(mapping.task_shape)]
        if not same_list(mapping.ranks, list(range(len(mapping.task_shape)))):
            items.append('ranks=[' + self(mapping.ranks) + ']')
        attr = self.add_attr("c", {"fn_type": "TaskMapping"})
        return f'repeat_map{attr}(' + doc_join(items, ', ') + ')'

    def visit_ComposedTaskMapping(self, mapping: ComposedTaskMapping):
        attr = self.add_attr("c", {"fn_type": "TaskMapping"})
        return f'compose_map{attr}(' + self(mapping.outer) + ',' + self(mapping.inner) + ')'

    def visit_StridesLayout(self, layout: StridesLayout):
        attr = self.add_attr("c", {"fn_type": "Layout"})

        if isinstance(layout, RowMajorLayout):
            return Text(f'row{attr}(') + self(layout.shape) + ')'
        elif isinstance(layout, ColumnMajorLayout):
            return Text(f'column{attr}(') + self(layout.shape) + ')'
        else:
            return Text(f'strides{attr}(') + self(layout.strides) + ')'
    
    # def get_strides_layout_attr

    def visit_SwizzleLayout(self, layout: SwizzleLayout):
        attr = self.add_attr("c", {"fn_type": "Layout"})
        items = [self(layout.base), Text('dim=') + self(layout.dim), Text('regards=') + self(layout.regards_dim)]
        if layout.log_step != 0:
            items.append(Text('log_step=') + self(layout.log_step))
        return Text(f'swizzle{attr}(') + doc_join(items, ', ') + ')'

    def visit_LocalLayout(self, layout: LocalLayout):
        attr = self.add_attr("c", {"fn_type": "Layout"})
        return Text(f'local{attr}(') + self(layout.shape) + ')'

    def visit_ComposedLayout(self, layout: ComposedLayout):
        attr = self.add_attr("c", {"fn_type": "Layout"})
        return Text(f'compose{attr}(') + self(layout.outer) + ', ' + self(layout.inner) + ')'

    def visit_ConcatLayout(self, layout: ConcatLayout):
        attr = self.add_attr("c", {"fn_type": "Layout"})
        return Text(f'concat{attr}(') + self(layout.lhs) + ', ' + self(layout.rhs) + ')'
    
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
        return Text('(!') + self(e.a) + ')'

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


def astext(obj: Node) -> str:
    if isinstance(obj, Node):
        printer = IRDumper()
        return str(printer(obj))
    else:
        raise ValueError()


from hidet.transforms.unify_global_objects import unify_global_objects_pass
from hidet.transforms.flatten_tensor_slice import flatten_tensor_slice_pass
from hidet.transforms.flatten_tensor_index import flatten_tensor_index_pass
from hidet.transforms.generate_launch_func import generate_launch_func_pass
from hidet.transforms.explicit_unroll import explicit_unroll_pass
from hidet.transforms.import_primitive_functions import import_primitive_functions_pass
from hidet.transforms.simplify_stmt import simplify_stmt_pass
from hidet.transforms.expand_let_expr import expand_let_expr_pass
from hidet.transforms.instantiate_symbols import instantiate_symbols_pass
from hidet.transforms.resolve_generic_primitive_function import resolve_primitive_func_pass
from hidet.transforms.inline_function import inline_function_pass
from hidet.transforms.add_explicit_cast import add_explicit_cast_pass
from hidet.transforms.inline_let_stmt import inline_let_stmt_pass
from hidet.transforms.rule_based_simplifier import rule_based_simplify_pass
from hidet.transforms.normalize_const_tensor import normalize_const_tensor_pass
from hidet.transforms.lower_task_mapping import lower_task_mapping_pass
from hidet.transforms.lower_protect_access import lower_protect_access_pass
from hidet.transforms.declare_to_let import declare_to_let_pass
from hidet.transforms.propagate_launch_bound import propagate_launch_bound_pass
from hidet.transforms.check_launch_configuration import check_launch_configuration_pass
from hidet.transforms.lower_special_cast import lower_special_cast_pass
from hidet.transforms.annotate_header_and_libs import annotate_header_and_libs_pass

# from hidet.graph.ops.softmax import SoftmaxTask
from hidet.graph.ops.matmul.matmul_f16 import MatmulF16Task
from hidet.graph.ops.matmul.batch_matmul import BatchMatmulTask
from hidet.graph.ops.softmax import SoftmaxTask
from hidet.graph.ops.attention.attention import AttnTask
from hidet.graph.ops.utils import input_like, tensor_input
from hidet import symbol_var

def get_matmul_task():
    s = symbol_var('s')
    a = tensor_input('a', 'float16', [s, 256])
    b = tensor_input('b', 'float16', [256, 512])
    task = MatmulF16Task(a, b)
    mods = task.implement_cuda('.')
    mod = mods[0]
    return mod

def get_bmatmul_task(mma_str='simt'):
    s = symbol_var('s')
    a = tensor_input('a', 'float16', [1, s, 256])
    b = tensor_input('b', 'float16', [1, 256, 256])
    task = BatchMatmulTask(a, b, mma_str)
    mods = task.implement_cuda('.')
    mod = mods[0]
    return mod

def get_softmax_task():
    a = tensor_input('a', 'float16', [1, 256])
    task = SoftmaxTask(a, 1)
    mod = task.implement_cuda('.')
    return mod

def get_attn_task():
    s = symbol_var('s')
    h = symbol_var('h')
    q = tensor_input('q', 'float16', [1, h, s, 64])
    k = tensor_input('k', 'float16', [1, h, s, 64])
    v = tensor_input('v', 'float16', [1, h, s, 64])
    task = AttnTask('attn', q, k, v, False)
    mod = task.implement_cuda('.')
    return mod[0]

def test_parser():
    from lark import Lark
    transforms = [
        # necessary passes
        lambda x: x,
        unify_global_objects_pass(),
        generate_launch_func_pass(),
        flatten_tensor_slice_pass(),
        lower_protect_access_pass(),
        lower_task_mapping_pass(),
        normalize_const_tensor_pass(),
        declare_to_let_pass(),
        rule_based_simplify_pass(),  # make ir more readable
        flatten_tensor_index_pass(),
        lower_special_cast_pass(),
        inline_function_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        propagate_launch_bound_pass(),
        add_explicit_cast_pass(),
        declare_to_let_pass(),
        instantiate_symbols_pass(),
        check_launch_configuration_pass(),
        # simplification
        expand_let_expr_pass(),
        inline_let_stmt_pass(),
        explicit_unroll_pass(),
        rule_based_simplify_pass(),
        inline_let_stmt_pass(),
        simplify_stmt_pass(),
        annotate_header_and_libs_pass(),
    ]

    with open('../hidet.lark') as f:
        hidet_grammar = f.read()
    parser = Lark(hidet_grammar, start='top_level', parser='lalr')

    for mod in [get_matmul_task(), get_bmatmul_task(), get_softmax_task(), get_attn_task()]:
        for t in transforms:
        
            mod = t(mod)
            text = astext(mod)
            print(text)
            try:
                tree = parser.parse(text)
                print(tree.pretty())
            except Exception as e:
                print(text)
                raise e


# test_parser()
mod = get_matmul_task()
text = astext(mod)
print(text)
# print(astext(mod))


import sys
from lark import Lark, Transformer, Token, Tree, Visitor
import logging
from lark import Lark, logger
logger.setLevel(logging.DEBUG)

with open('../hidet.lark') as f:
    hidet_grammar = f.read()

parser = Lark(hidet_grammar, start='top_level', debug=True, parser='lalr')
tree = parser.parse(text)
print(tree.pretty())


# %%

def construct_pair(tree):
    assert tree.data.value == 'pair'
    return tree.children[0], tree.children[1]

def preprocess_symbolvar(tree: Tree):
    assert tree.data.value == 'top_level'
    attr = tree.children[0]
    assert attr.data.value == 'attribute'
    attr_dict = attr.children[1].children
    for tree in attr_dict:
        name, value = construct_pair(tree)
        if str(name)[1:-1] == 'symbols':
            symbol_var = {}
            for symbol in value.children:
                symbol_name, symbol_type = construct_pair(symbol)
                symbol_var[str(symbol_name)[1:-1]] = str(symbol_type)[1:-1]
            return symbol_var
    return None

def construct_symbols(d: Dict[str, str]) -> Dict[str, SymbolVar]:
    symbols = {}
    for k, v in d.items():
        symbols[k] = symbol_var(k, v)
    return symbols

class PreProcessTree(Visitor):
    def __init__(self, symbols: Dict[str, SymbolVar]):
        self.attributes: Dict[str, dict] = {}
        self.symbols: Dict[str, SymbolVar] = symbols
        self.funcs: Set[str] = set()
    
    def attribute(self, tree):
        name = tree.children[0].children[0]
        value = tree.children[1]
        self.attributes[str(name)] = AttrConstructor(self.symbols).transform(value)
    
    def fn_name(self, name):
        self.funcs.add(str(name.children[0]))

def resolve_binary_op(op, a, b):
    if op == '&&':
        return ir.logical_and(a, b)
    elif op == '||':
        return ir.logical_or(a, b)
    return eval(f'a {op} b')

def resolve_unary_op(op, a):
    if op == '!':
        return ir.logical_not(a)
    return eval(f'{op} a')

class AttrConstructor(Transformer):
    def __init__(self, symbolvar: Dict[str, SymbolVar]) -> None:
        super().__init__()
        self.symboltable = symbolvar

    def STRING(self, s):
        # (s,) = s
        return s[1:-1]
    def INT(self, n):
        # (n,) = n
        return int(n)
    
    def SIGNED_NUMBER(self, n):
        # (n,) = n
        return float(n)
    
    def symbol_var(self, var):
        (var,) = var
        return self.symboltable[var]
    
    def binary_op(self, op):
        a, op, b = op
        return resolve_binary_op(op, a, b)
    
    def unary_op(self, op):
        op, a = op
        return resolve_unary_op(op, a)
    
    def if_then_else_expr(self, expr):
        cond, then_expr, else_expr = expr
        return ir.expr.if_then_else(cond, then_expr, else_expr)

    list = list
    pair = tuple
    dict = dict

    none = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False

class IRConstructor(Transformer):
    def __init__(self, preprocessor: PreProcessTree):
        super().__init__()
        self.preprocessor = preprocessor
        self.symboltable = preprocessor.symbols
        self.attributes = preprocessor.attributes
        self.funcs = preprocessor.funcs

    def STRING(self, s):
        # (s,) = s
        return s[1:-1]
    
    def INT(self, n):
        # (n,) = n
        return int(n)
    
    def SIGNED_NUMBER(self, n):
        # (n,) = n
        return float(n)
    
    def IDENT(self, var):
        # (var,) = var
        var = str(var)

        return var
    
    def symbol_var(self, var):
        (var,) = var
        return self.symboltable[var]
    
    def binary_op(self, op):
        a, op, b = op
        print(a, b)
        return resolve_binary_op(op, a, b)
    
    def unary_op(self, op):
        op, a = op
        return resolve_unary_op(op, a)
    
    def if_then_else_expr(self, expr):
        cond, then_expr, else_expr = expr
        return ir.expr.if_then_else(cond, then_expr, else_expr)
    
    def top_level(self, node):
        return node[1]

class IRConstructor:
    def __init__(self, preprocessor: PreProcessTree, pass_through=True):
        self.pass_through = pass_through
        self.preprocessor = preprocessor
        self.symboltable = preprocessor.symbols
        self.attributes = preprocessor.attributes
        self.funcs = preprocessor.funcs

        self.var_table: Dict[str, Var] = {}

    def __call__(self, node) -> Node:
        return self.visit(node)
    
    def visit(self, node) -> Node:
        if isinstance(node, Tree):
            if isinstance(node.data, str):
                # alias
                method_name = 'visit_' + node.data
            else:
                # rule
                assert node.data.type == 'RULE'
                method_name = 'visit_' + node.data.value
            method = getattr(self, method_name, None)
            if method is None:
                if not self.pass_through:
                    raise NotImplementedError("No method {} for {}".format(method_name, node.data))
                children = [self.visit(child) for child in node.children]
                return Tree(node.data, children, meta=node.meta)
            else:
                return method(node.children)

        elif isinstance(node, Token):
            method_name = 'visit_' + node.type
            method = getattr(self, method_name, None)
            if method is None:
                if not self.pass_through:
                    raise NotImplementedError("No method {} for {}".format(method_name, node.type))
                return node
            else:
                return method(node.value)
        elif node is None:
            return node
        elif isinstance(node, (str, int, float, Node)):
            return node
        else:
            if not self.pass_through:
                raise NotImplementedError("Unknown node type: {}".format(type(node)))
            else:
                print(type(node), node)
    
    def visit_STRING(self, value):
        return str(value[1:-1])

    def visit_INT(self, value):
        return int(value)

    def visit_SIGNED_FLOAT(self, value):
        return float(value)
    
    def visit_true(self, _):
        return True
    
    def visit_false(self, _):
        return False
    
    def visit_none(self, _):
        return None
    
    def visit_pair(self, node):
        return self.visit(node[0]), self.visit(node[1])
    
    def visit_dict(self, node):
        pairs = [self.visit(child) for child in node]
        return {key: value for key, value in pairs}
    
    def visit_list(self, node):
        return [self.visit(child) for child in node]

    def visit_top_level(self, children):
        # for child in children[:-1]:
        #     if child.data.value == 'attribute':
        #         self.visit(child)
            
        return self.visit(children[-1])
    
    def visit_attribute_name(self, name):
        return str(name[0])

    def visit_data_type(self, type):
        type_name = str(type[0])
        if type_name == 'void':
            return hidet.ir.type.void
        
        return ir.data_type(type_name)
    
    def visit_unary_op(self, node):
        op, expr = node
        try:
            return resolve_unary_op(op, self.visit(expr))
        except Exception as e:
            return node
    
    def visit_binary_op(self, node):
        a, op, b = node
        try:
            return resolve_binary_op(op, self.visit(a), self.visit(b))
        except Exception as e:
            return node
    
    def visit_fn_name(self, node):
        name_to_taskmap = {
            'compose_map': ComposedTaskMapping,
            'repeat_map': RepeatTaskMapping,
            'spatial_map': SpatialTaskMapping
        }
        name_to_layout = {
            'row': RowMajorLayout,
            'column': ColumnMajorLayout,
            'strides': StridesLayout,
            'swizzle': SwizzleLayout,
            'local': LocalLayout,
            'compose': ComposedLayout,
            'concat': ConcatLayout
        }
        name = str(node[0])
        attr_name = self(node[1])
        # print(type(attr), attr)
        if attr_name is not None:
            attr = self.attributes[attr_name]
            assert isinstance(attr, dict)
            if "fn_type" in attr:
                fn_type = attr["fn_type"]
                if fn_type == "TaskMapping":
                    return name_to_taskmap[name]
                elif fn_type == "Layout":
                    return name_to_layout[name]
                else:
                    raise RuntimeError(f"Unknown fn_type {fn_type}")
        if is_primitive_function(name):
            return lookup_primitive_function(name)
        if name in self.var_table:
            return self.var_table[name]
        if name in __builtins__:
            return __builtins__[name]
        
        raise RuntimeError(f"cannot find function {name}")
    
    def visit_name(self, name):
        return str(name[0])
    
    
    # def visit_module(self, children):

    #     return ir.Module(self.visit(children[0]), self.visit(children[1]))

symbol_table = construct_symbols(preprocess_symbolvar(tree))
preprocess = PreProcessTree(symbol_table)
preprocess.visit(tree)
IRConstructor(preprocess)(tree)


# %%
