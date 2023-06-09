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
from typing import Optional, List, Tuple, Dict, Union
import os
import numpy as np
from hidet.ir.dialects.pattern import PlaceholderExpr
from hidet.ir import dtypes
from hidet.ir.node import Node
from hidet.ir.type import DataType, PointerType, TensorPointerType, ReferenceType, TensorType, FuncType
from hidet.ir.type import VoidType, StringType, ArrayType
from hidet.ir.expr import Var, Add, Sub, Multiply, Div, Mod, FloorDiv, LessThan, Neg, NotEqual, Equal, LogicalAnd
from hidet.ir.expr import LogicalOr, LogicalNot, BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, LeftShift, RightShift
from hidet.ir.expr import IfThenElse, Cast, Address, Reference, Dereference, Call, Let, Constant, TensorSlice, convert
from hidet.ir.expr import TensorElement
from hidet.ir.stmt import DeclareScope, DeclareStmt, EvaluateStmt, BufferStoreStmt, AssignStmt, LetStmt, ForStmt
from hidet.ir.stmt import LaunchKernelStmt
from hidet.ir.stmt import ForMappingStmt, WhileStmt, BreakStmt, ContinueStmt, IfStmt, ReturnStmt, AssertStmt, AsmStmt
from hidet.ir.stmt import BlackBoxStmt, SeqStmt
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.compute import TensorNode, ScalarNode
from hidet.ir.functors import ModuleFunctor, StmtFunctor, ExprFunctor, TypeFunctor
from hidet.ir.tools import TypeInfer
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.ir.utils.call_graph import CallGraph
from hidet.utils.namer import Namer
from hidet.utils import prod
from hidet.ir.primitives import is_primitive_function, lookup_primitive_function


class Codegen(ModuleFunctor, StmtFunctor, ExprFunctor, TypeFunctor):
    def __init__(self):
        super().__init__()
        self.ir_module: Optional[IRModule] = None
        self.namer = Namer()
        self.type_infer = TypeInfer()

        self.require_immintrin = False
        self.require_complex = False
        self.require_fp16 = False
        self.require_bf16 = False
        self.require_tf32 = False

    def __call__(self, node) -> Doc:
        return self.visit(node)

    def canonize_funcname(self, name: str):
        items = name.split('.')
        if len(items) == 1:
            if self.ir_module.namespace != '':
                return items[0]
            else:
                return 'hidet_' + items[0]  # only add 'hidet_' prefix when no namespace
        elif len(items) == 2:
            return items[0] + '::' + items[1]
        else:
            raise ValueError('Cannot recognize function name {}'.format(name))

    def scalar_literal(self, value, dtype: DataType):
        if dtype == dtypes.boolean:
            ret = 'true' if value else 'false'
        elif dtype == dtypes.float64:
            ret = '{}'.format(float(value))
        elif dtype == dtypes.float32:
            ret = '{}f'.format(float(value))
        elif dtype == dtypes.float16:
            ret = '(half){}f'.format(float(value))
        elif dtype == dtypes.tfloat32:
            ret = '(float){}f'.format(float(value))
        elif dtype == dtypes.bfloat16:
            ret = '(bfloat16_t){}f'.format(float(value))
        elif dtype == dtypes.int64:
            ret = 'int64_t({}ll)'.format(int(value))
        elif dtype == dtypes.int32:
            ret = '{}'.format(int(value))
        elif dtype == dtypes.int16:
            ret = 'int16_t({})'.format(int(value))
        elif dtype == dtypes.int8:
            ret = 'int8_t({})'.format(int(value))
        elif dtype == dtypes.uint64:
            ret = 'uint64_t({}ull)'.format(int(value))
        elif dtype == dtypes.uint32:
            ret = 'uint32_t({}u)'.format(int(value))
        elif dtype == dtypes.uint16:
            ret = 'uint16_t({})'.format(int(value))
        elif dtype == dtypes.uint8:
            ret = 'uint8_t({})'.format(int(value))
        elif dtype.is_complex():
            if not isinstance(value, complex):
                raise ValueError('Cannot recognize scalar literal {} with dtype {}'.format(value, dtype))
            if dtype == dtypes.complex64:
                ret = 'complex64_t({}, {})'.format(value.real, value.imag)
            elif dtype == dtypes.complex128:
                ret = 'complex128_t({}, {})'.format(value.real, value.imag)
            else:
                raise NotImplementedError('Cannot recognize scalar literal {} with dtype {}'.format(value, dtype))
        else:
            raise NotImplementedError('Cannot recognize scalar literal {} with dtype {}'.format(value, dtype))
        return Text(ret)

    def param_declare(self, v: Var):
        v_type = v.type
        name_doc = self(v)
        if isinstance(v_type, DataType):
            dtype_doc = self(v_type)
            return dtype_doc + ' ' + name_doc
        elif isinstance(v_type, PointerType):
            if len(v_type.specifiers) > 0:
                attr_doc = doc_join([self(attr) for attr in v_type.specifiers], sep=' ') + ' '
            else:
                attr_doc = Doc()
            dtype = v_type.base_type
            base_type_doc = self(dtype)
            if v_type.use_bracket:
                return attr_doc + base_type_doc + ' ' + name_doc + '[]'
            else:
                return attr_doc + base_type_doc + ' *' + ' __restrict__ ' + name_doc
        elif isinstance(v_type, TensorPointerType):
            dtype = v_type.tensor_type.dtype
            base_type_doc = self(dtype)
            return base_type_doc + ' *' + ' __restrict__ ' + name_doc
        elif isinstance(v_type, ReferenceType):
            if isinstance(v_type.base_type, DataType):
                base_type_doc = self(v_type.base_type)
                return base_type_doc + ' &' + name_doc
            else:
                raise NotImplementedError()
        elif isinstance(v_type, TensorType):
            dtype = v_type.dtype
            base_type_doc = self(dtype)
            return base_type_doc + ' *' + ' __restrict__ ' + name_doc
        elif isinstance(v_type, ArrayType):
            base_type_doc = self(v_type.base_type)
            return base_type_doc + ' ' + name_doc + '[' + self(v_type.size) + ']'
        else:
            raise ValueError()

    def local_var_declare(self, v: Var):
        v_type = v.type
        name_doc = self(v)
        if isinstance(v_type, DataType):
            dtype_doc = self(v_type)
            return dtype_doc + ' ' + name_doc
        elif isinstance(v_type, TensorType):
            dtype_doc = self(v_type.dtype)
            shape_doc = Doc()
            for s in v_type.shape:
                shape_doc += '[' + self(s) + ']'
            return dtype_doc + ' ' + name_doc + shape_doc
        elif isinstance(v_type, PointerType):
            if len(v_type.specifiers) > 0:
                attr_doc = doc_join([self(attr) for attr in v_type.specifiers], sep=' ') + ' '
            else:
                attr_doc = Doc()
            base_type_doc = self(v_type.base_type)
            if v_type.use_bracket:
                return attr_doc + base_type_doc + ' ' + name_doc + '[]'
            else:
                return attr_doc + base_type_doc + ' *' + name_doc
        elif isinstance(v_type, TensorPointerType):
            dtype_doc = self(v_type.tensor_type.dtype)
            return dtype_doc + ' *' + name_doc
        elif isinstance(v_type, FuncType):
            return_type_doc = self(v_type.ret_type)
            args_doc = doc_join([self(param_type) for param_type in v_type.param_types], sep=', ')
            return return_type_doc + ' (*' + name_doc + ')(' + args_doc + ')'
        elif isinstance(v_type, ArrayType):
            if isinstance(v_type.base_type, FuncType):
                return_type_doc = self(v_type.base_type.ret_type)
                args_doc = doc_join([self(param_type) for param_type in v_type.base_type.param_types], sep=', ')
                return return_type_doc + ' (*' + name_doc + '[' + self(v_type.size) + '])(' + args_doc + ')'
            else:
                base_type_doc = self(v_type.base_type)
                return base_type_doc + ' ' + name_doc + '[' + self(v_type.size) + ']'
        else:
            assert False

    def require_headers(self) -> Doc:
        return Doc()

    def visit(self, node):
        if isinstance(node, Doc):
            return node
        else:
            return super().visit(node)

    def visit_List(self, lst: List):
        return doc_join([self(v) for v in lst], ', ')

    def visit_Tuple(self, tp: Tuple):
        return doc_join([self(v) for v in tp], ', ')

    def visit_Dict(self, dct: Dict):
        raise RuntimeError('Dict is not supported in code generation')

    def visit_PyConstant(self, c: Union[str, int, float, None]):
        if c is None:
            raise RuntimeError('None encountered during code generation')
        return Text(str(c))

    def declare_function(self, func_name, func_type: FuncType):
        doc = Doc()

        # return type
        doc += self(func_type.ret_type)

        # func name
        doc += ' ' + func_name

        # parameters
        doc += '('
        param_docs = []
        for param_type in func_type.param_types:
            param_docs.append(self.param_declare(Var('v', param_type)))
        doc += doc_join(param_docs, Text(', '))
        doc += ');'
        return doc

    def visit_IRModule(self, module: IRModule) -> Doc:
        self.ir_module = module
        doc = Doc()

        for name, func_var in module.extern_functions.items():
            items = name.split('.')
            assert len(items) <= 2
            if len(items) == 1:
                doc += self.declare_function(name, func_var.type) + NewLine()
            else:
                doc += 'namespace ' + items[0] + ' {' + NewLine()
                doc += '  ' + self.declare_function(items[1], func_var.type) + NewLine()
                doc += '} ' + NewLine()

        if module.namespace != '':
            doc += 'namespace ' + module.namespace + ' {' + NewLine()

        # define global variables
        for name, var in module.global_vars.items():
            if name in module.functions:
                continue
            doc += self.local_var_declare(var) + ';' + NewLine()

        # define functions
        call_graph = CallGraph(module)
        for node in call_graph.reversed_order:
            doc += self(node.func) + NewLine()

        if module.namespace != '':
            doc += NewLine()
            doc += '}  // namespace ' + module.namespace + NewLine()

        doc = self.require_headers() + doc

        return doc

    def visit_Function(self, func: Function) -> Doc:
        raise NotImplementedError()

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

    def visit_Neg(self, e: Neg):
        return '(-' + self(e.a) + ')'

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
            raise ValueError('The protected reading of tensor element should be lowered in lower_protect_access pass.')
        base_doc = self(e.base)
        index_doc = doc_join(['[' + self(idx) + ']' for idx in e.indices], '')
        if isinstance(e.base, Address):
            return Text('(') + base_doc + Text(')') + index_doc
        else:
            return base_doc + index_doc

    def visit_IfThenElse(self, e: IfThenElse):
        return '(' + self(e.cond) + ' ? ' + self(e.then_expr) + ' : ' + self(e.else_expr) + ')'

    def visit_Cast(self, e: Cast):
        src_dtype = self.type_infer(e.expr)
        dst_dtype = e.target_type
        if isinstance(src_dtype, DataType) and isinstance(dst_dtype, DataType) and src_dtype == dtypes.float16:
            # in cuda, cuda_fp16.h only defines the half struct with conversion operators for the types like float,
            # short, int, unsigned int, long long, unsigned long long, but not for the types like int8_t, uint8_t,
            # int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, so we need to cast them here.
            if dst_dtype == dtypes.int64:
                return '(int64_t)((long long)(' + self(e.expr) + '))'
            elif dst_dtype == dtypes.uint64:
                return '(uint64_t)((unsigned long long)(' + self(e.expr) + '))'
            elif dst_dtype == dtypes.int32:
                return '(int32_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.uint32:
                return '(uint32_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.int16:
                return '(int16_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.uint16:
                return '(uint16_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.int8:
                return '(int8_t)(short)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.uint8:
                return '(uint8_t)(unsigned short)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.boolean:
                return '(bool)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.float32:
                return '(float)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.float64:
                return '(double)(' + self(e.expr) + ')'
            else:
                return Text('((') + self.visit(e.target_type) + ')(' + self(e.expr) + '))'
        else:
            return Text('((') + self.visit(e.target_type) + ')(' + self(e.expr) + '))'

    def visit_Address(self, e: Address):
        return Text('&') + self.visit(e.expr)

    def visit_Reference(self, e: Reference):
        raise ValueError()

    def visit_Dereference(self, e: Dereference):
        return Text('*') + self(e.expr)

    def visit_Call(self, e: Call):
        func_name: str = e.func_var.name if e.func_var.name else e.func_var.hint
        assert isinstance(func_name, str)
        if func_name in self.ir_module.functions:
            func = self.ir_module.functions[func_name]
            func_name = self.canonize_funcname(func_name)
            if func.kind == 'cuda_kernel':
                raise RuntimeError('Call to cuda kernel should be lowered to LaunchKernelStmt.')
            param_doc = Text('(') + doc_join([self(arg) for arg in e.args], Text(', ')) + ')'
            return func_name + param_doc
        elif is_primitive_function(func_name):
            entry = lookup_primitive_function(func_name)
            if entry.function is not None:
                msg = (
                    f"Please use import_primitive_functions pass to import primitive function first: {entry.name}, "
                    f"functions in current module:\n{list(self.ir_module.functions.keys())}."
                )
                raise ValueError(msg)
            if entry.generic:
                msg = (
                    "Please use resolve_generic_primitive_function pass to lower "
                    "the generic primitive function {}.".format(entry.name)
                )
                raise ValueError(msg)
            # system-provided function, do not canonize the func name
            return entry.codegen_name + (Text('(') + doc_join([self(arg) for arg in e.args], Text(', ')) + ')')
        else:
            # call a local function pointer variable
            return self(e.func_var) + Text('(') + doc_join([self(arg) for arg in e.args], Text(', ')) + Text(')')

    def visit_Let(self, e: Let):
        raise ValueError("please run 'expand_let_expr' pass before codegen")

    def visit_Var(self, e: Var):
        cast2int = {'threadIdx.x', 'threadIdx.y', 'threadIdx.z', 'blockIdx.x', 'blockIdx.y', 'blockIdx.z'}
        name = self.namer.get_name(e)
        if name in cast2int:
            return Text(f'(int){name}')
        else:
            if isinstance(e.type, FuncType):
                name = self.canonize_funcname(name)
            return Text(name)

    def visit_Constant(self, e: Constant):
        if e.is_string():
            return Text('"{}"'.format(e.value))
        elif e.is_scalar():
            return self.scalar_literal(e.value, e.type)
        elif e.is_tensor():
            dtype = e.type.dtype
            items = [self.scalar_literal(v, dtype) for v in np.array(e.value).flatten()]
            return '{' + doc_join(items, ', ') + '}'
        elif isinstance(e.type, PointerType):
            return '(' + self(e.type) + ')' + str(int(e.value))
        else:
            raise ValueError("invalid constant type: {}".format(e))

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        doc = NewLine()
        if stmt.is_static:
            doc += 'static '
        if stmt.scope != DeclareScope.Default:
            scope2specifier = {
                DeclareScope.Shared: '__shared__ ',
                DeclareScope.Global: '__global__ ',
                DeclareScope.Register: '',  # we can not force nvcc to use register, but it will do so if possible
            }
            doc += scope2specifier[stmt.scope]
        doc += self.local_var_declare(stmt.var)
        if stmt.init is not None:
            doc += ' = ' + self(stmt.init)
        return doc + ';'

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        return NewLine() + self(stmt.expr) + ';'

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        if stmt.protected:
            raise ValueError('The protected writing of tensor element should be lowered in lower_protect_access pass.')
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
            doc += NewLine() + self.local_var_declare(bind_var) + ' = ' + self(bind_value) + ';'
            # doc += NewLine() + self(bind_var.type) + ' ' + self(bind_var) + ' = ' + self(bind_value) + ';'
        doc += self(stmt.body)
        return doc

    def visit_ForStmt(self, stmt: ForStmt):
        v = stmt.loop_var
        init_doc = self(v.type) + ' ' + self(v) + ' = ' + self(convert(0))
        cond_doc = self(v < stmt.extent)
        update_doc = self(v) + ' = ' + self(v + 1)
        doc = Text('')
        if stmt.attr.unroll:
            assert not stmt.attr.unroll_explicit, 'explicit_unroll should be lowered before codegen'
            if stmt.attr.unroll_factor:
                doc += NewLine() + '#pragma unroll {}'.format(stmt.attr.unroll_factor)
            else:
                doc += NewLine() + '#pragma unroll'
        elif stmt.attr.parallel:
            if stmt.attr.parallel_threads:
                doc += NewLine() + '#pragma omp parallel for schedule(dynamic) num_threads({})'.format(
                    stmt.attr.parallel_threads
                )
            else:
                doc += NewLine() + '#pragma omp parallel for'
        doc += NewLine() + Text('for (') + init_doc + '; ' + cond_doc + '; ' + update_doc + ') '
        body_doc = self(stmt.body)
        doc += Text('{') + body_doc.indent() + NewLine() + Text('} ')
        return doc

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        raise ValueError('ForTaskStmt should be lowered to ForStmt in lower_task_mapping pass before code generation.')

    def visit_WhileStmt(self, stmt: WhileStmt):
        doc = NewLine() + Text('while (') + self(stmt.cond) + ')'
        body_doc = self(stmt.body)
        doc += Text(' {') + body_doc.indent() + NewLine() + Text('} ')
        return doc

    def visit_BreakStmt(self, stmt: BreakStmt):
        return NewLine() + 'break;'

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        return NewLine() + 'continue;'

    def visit_IfStmt(self, stmt: IfStmt):
        cond_doc = self(stmt.cond)
        if not (len(cond_doc.docs) > 0 and isinstance(cond_doc.docs[0], str) and cond_doc.docs[0].startswith('(')):
            cond_doc = Text('(') + cond_doc + ')'
        doc = NewLine() + Text('if ') + cond_doc + ' '
        doc += Text('{') + self(stmt.then_body).indent() + NewLine() + Text('} ')
        if stmt.else_body:
            doc += Text('else ')
            doc += Text('{') + self(stmt.else_body).indent() + NewLine() + Text('} ')
        return doc

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        doc = Doc()
        doc += NewLine() + 'return'
        if stmt.ret_value is not None:
            doc += ' ' + self(stmt.ret_value)
        doc += ';'
        return doc

    def visit_AssertStmt(self, stmt: AssertStmt):
        if stmt.msg is not None:
            msg = stmt.msg.replace("\n", "")
        else:
            msg = 'assertion failed!'
        
        doc = NewLine() + Text(f'hidet_set_last_error("{msg}");') + NewLine()
        doc += Text('return;')
        
        doc = NewLine() + Text('if (!') + self(stmt.cond) + Text(') {') + doc.indent() + NewLine() + Text('}')
        return doc
        
    def visit_AsmStmt(self, stmt: AsmStmt):
        volatile_doc = 'volatile ' if stmt.is_volatile else ''
        template_doc = f'"{Text(stmt.template_string)}"'
        output_docs = []
        for label, expr in zip(stmt.output_labels, stmt.output_exprs):
            output_docs.append(Text(f'"{label}"') + '(' + self(expr) + ')')
        input_docs = []
        for label, expr in zip(stmt.input_labels, stmt.input_exprs):
            input_docs.append(Text(f'"{label}"') + '(' + self(expr) + ')')
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
        assert isinstance(stmt.func_var, Var)
        return NewLine() + Text('{}<<<dim3({}), dim3({}), {}, {}>>>({});').format(
            self.canonize_funcname(stmt.func_var.name),
            self(stmt.grid_dim),
            self(stmt.block_dim),
            self(stmt.shared_mem_bytes),
            Text("(cudaStream_t)get_cuda_stream()"),
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

    def visit_DataType(self, t: DataType):
        scalar_type_map = {
            'bool': 'bool',
            'uint8': 'uint8_t',
            'uint16': 'uint16_t',
            'uint32': 'uint32_t',
            'uint64': 'uint64_t',
            'int8': 'int8_t',
            'int16': 'int16_t',
            'int32': 'int32_t',
            'int64': 'int64_t',
            'float16': 'half',
            'float32': 'float',
            'float64': 'double',
            'bfloat16': 'bfloat16_t',
            'tfloat32': 'tfloat32_t',
            'complex64': 'complex64_t',
            'complex128': 'complex128_t',
            'float32x4': '__m128',
            'float32x8': '__m256',
        }

        self.require_complex = self.require_complex or t.name in ['complex64', 'complex128']
        self.require_immintrin = self.require_immintrin or t.name in ['float32x4', 'float32x8']
        self.require_bf16 = self.require_bf16 or t.name == 'bfloat16'
        self.require_fp16 = self.require_fp16 or t.name == 'float16'
        self.require_tf32 = self.require_tf32 or t.name == 'tfloat32'

        return Text(scalar_type_map[t.name])

    def visit_TensorType(self, t: TensorType):
        raise ValueError()

    def visit_ArrayType(self, t: ArrayType):
        raise ValueError()

    def visit_PointerType(self, t: PointerType):
        return self(t.base_type) + Text('*')

    def visit_TensorPointerType(self, t: TensorPointerType):
        return self(t.tensor_type.dtype) + Text('*')

    def visit_ReferenceType(self, t: ReferenceType):
        raise ValueError()

    def visit_VoidType(self, t: VoidType):
        return Text('void')

    def visit_StringType(self, t: StringType):
        return Text('char*')

    def visit_FuncType(self, t: FuncType):
        ret_type_doc = self(t.ret_type)
        param_type_docs = [self(p) for p in t.param_types]
        return ret_type_doc + Text(' (*)( ') + doc_join(param_type_docs, ', ') + Text(')')

    # the following expressions should not remain to codegen
    def visit_TensorSlice(self, e: TensorSlice):
        raise ValueError()

    def visit_ScalarNode(self, e: ScalarNode):
        raise ValueError()

    def visit_TensorNode(self, e: TensorNode):
        raise ValueError()

    def visit_PlaceholderExpr(self, e: PlaceholderExpr):
        raise ValueError()

    def visit_NotDispatchedNode(self, n: Node):
        raise ValueError()


class CUDACodegen(Codegen):
    def require_headers(self) -> Doc:
        doc = Doc()
        doc += Text('#include <stdint.h>') + NewLine()

        if self.require_immintrin:
            doc += Text('#include <immintrin.h>') + NewLine()
        if self.require_fp16:
            doc += Text('#include <cuda_fp16.h>') + NewLine()
        if self.require_bf16:
            doc += Text('#include <cuda_bf16.h>') + NewLine()
        doc += Text('#include <hidet/runtime/symbols.h>') + NewLine()
        doc += Text('#include <hidet/runtime/memory_planner.h>') + NewLine()
        doc += Text('#include <hidet/runtime/cpu/context.h>') + NewLine()
        doc += Text('#include <hidet/runtime/cuda/complex.h>') + NewLine()
        doc += Text('#include <hidet/runtime/cuda/context.h>') + NewLine()
        doc += Text("#include <hidet/runtime/logging.h>") + NewLine()


        if self.require_tf32:
            # nvcc use float to 'store' tfloat32 data
            doc += Text('typedef float tfloat32_t;') + NewLine()

        if self.require_bf16:
            doc += Text('typedef __nv_bfloat16 bfloat16_t;') + NewLine()

        doc += NewLine()
        return doc

    def visit_Function(self, func: Function) -> Doc:
        self.namer.clear()

        doc = NewLine()

        # ret
        if func.kind == 'cuda_kernel':
            doc += 'static __global__ '
        elif func.kind == 'cuda_internal':
            doc += 'static __device__ __forceinline__ '
        elif func.kind == 'cpu_kernel':
            doc += 'static '
        elif func.kind == 'cpu_internal':
            doc += 'static __forceinline__ '
        elif func.kind == 'public':
            if self.ir_module.namespace == '':
                doc += 'DLL '
        else:
            raise ValueError(f'Unknown function kind: {func.kind}')

        doc += self(func.ret_type)

        # launch bound for grid worker
        if func.kind == 'cuda_kernel':
            block_dim = func.attrs['cuda.block_dim']
            if isinstance(block_dim, list):
                block_dim = prod(block_dim)
            if isinstance(block_dim, (Constant, int)):
                if 'cuda.min_blocks' in func.attrs:
                    min_blocks = func.attrs['cuda.min_blocks']
                    if isinstance(min_blocks, (Constant, int)):
                        doc += f' __launch_bounds__({block_dim}, {min_blocks})'
                    else:
                        doc += f' __launch_bounds__({block_dim})'
                else:
                    doc += f' __launch_bounds__({block_dim})'

        # func name
        canonized_func_name = self.canonize_funcname(func.name)
        doc += ' ' + canonized_func_name

        # parameters
        doc += '('
        param_docs = []
        for param in func.params:
            param_docs.append(self.param_declare(param))
        doc += doc_join(param_docs, Text(', '))
        doc += ') {'

        # body
        doc += self(func.body).indent()

        doc += NewLine() + '}'

        return doc


class CPUCodegen(Codegen):
    def require_headers(self) -> Doc:
        doc = Doc()
        doc += Text('#include <stdint.h>') + NewLine()
        doc += Text('#include <math.h>') + NewLine()
        if self.require_immintrin:
            doc += Text('#include <immintrin.h>') + NewLine()
        doc += Text('#include <hidet/runtime/symbols.h>') + NewLine()
        doc += Text('#include <hidet/runtime/memory_planner.h>') + NewLine()
        doc += Text('#include <hidet/runtime/cpu/context.h>') + NewLine()
        doc += Text('#include <hidet/runtime/cpu/float32.h>') + NewLine()
        doc += Text("#include <hidet/runtime/logging.h>") + NewLine()

        if self.require_complex:
            doc += Text('#include <hidet/runtime/cpu/complex.h>') + NewLine()
        if self.require_fp16:
            doc += Text('#include <hidet/runtime/cpu/float16.h>') + NewLine()
        if self.require_bf16:
            doc += Text('#include <hidet/runtime/cpu/bfloat16.h>') + NewLine()
        if self.require_tf32:
            doc += Text('typedef float tfloat32_t;') + NewLine()
        doc += NewLine()
        return doc

    def visit_Function(self, func: Function) -> Doc:
        self.namer.clear()
        doc = NewLine()

        if func.kind == 'public':
            if self.ir_module.namespace == '':
                doc += 'DLL '
            else:
                doc += ''
        elif func.kind in ['cpu_kernel', 'cpu_internal']:
            doc += 'static '
        else:
            raise ValueError(f'Unsupported function kind: {func.kind}')

        doc += self(func.ret_type)

        # func name
        canonized_func_name = self.canonize_funcname(func.name)
        doc += ' ' + canonized_func_name

        # parameters
        doc += '('
        param_docs = []
        for param in func.params:
            param_docs.append(self.param_declare(param))
        doc += doc_join(param_docs, Text(', '))
        doc += ') {'

        # body
        doc += self(func.body).indent()

        doc += NewLine() + '}'

        return doc


def codegen(ir_module: IRModule, src_out_path: str, target: str) -> str:
    if target == 'cuda':
        gen = CUDACodegen()
    elif target == 'cpu':
        gen = CPUCodegen()
    else:
        raise ValueError(f'Unknown target: {target}')

    doc = gen(ir_module)
    code = str(doc)
    if src_out_path is not None:
        dir_path = os.path.dirname(src_out_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(src_out_path, 'w') as f:
            f.write(code)
    return code
