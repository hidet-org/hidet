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
from typing import Optional
import os
import numpy as np
from hidet.ir.dialects.pattern import AnyExpr
from hidet.ir import dtypes
from hidet.ir.type import DataType, PointerType, TensorPointerType, ReferenceType, TensorType, TypeNode, FuncType
from hidet.ir.type import VoidType
from hidet.ir.expr import (
    Var,
    Expr,
    Add,
    Sub,
    Multiply,
    Div,
    Mod,
    FloorDiv,
    LessThan,
    Neg,
    NotEqual,
    Equal,
    LogicalAnd,
    LogicalOr,
)
from hidet.ir.expr import (
    LogicalNot,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNot,
    LeftShift,
    RightShift,
    TensorElement,
)
from hidet.ir.expr import IfThenElse, Cast, Address, Reference, Dereference, Call, Let, Constant, TensorSlice, convert
from hidet.ir.stmt import Stmt, DeclareScope, DeclareStmt, EvaluateStmt, BufferStoreStmt, AssignStmt, LetStmt, ForStmt
from hidet.ir.stmt import LaunchKernelStmt
from hidet.ir.stmt import ForTaskStmt, WhileStmt, BreakStmt, ContinueStmt, IfStmt, ReturnStmt, AssertStmt, AsmStmt
from hidet.ir.stmt import BlackBoxStmt, SeqStmt
from hidet.ir.func import IRModule, Function
from hidet.ir.compute import TensorNode, ScalarNode
from hidet.ir.functors import StmtExprFunctor, TypeFunctor, TypeInfer
from hidet.utils.doc import Doc, NewLine, Text, doc_join
from hidet.ir.utils.call_graph import CallGraph
from hidet.utils.namer import Namer
from hidet.utils import prod
from hidet.ir.primitives import is_primitive_function, lookup_primitive_function


class Codegen(StmtExprFunctor, TypeFunctor):
    def __init__(self):
        super().__init__()
        self.func_name_map = {}
        self.ir_module: Optional[IRModule] = None
        self.namer = Namer()
        self.type_infer = TypeInfer()

    @staticmethod
    def canonize_funcname(name: str):
        return 'hidet_' + name.replace('.', '_')

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
            # dtype_doc = self(v_type.scalar_type)
            # name_doc = self(v)
            # shape_doc = Doc()
            # for s in v_type.shape:
            #     shape_doc += '[' + self(s) + ']'
            # return dtype_doc + ' ' + '__restrict__' + name_doc + shape_doc
        else:
            raise ValueError()

    def local_var_declare(self, v: Var):
        v_type = v.type
        if isinstance(v_type, DataType):
            dtype_doc = self(v_type)
            name_doc = self(v)
            return dtype_doc + ' ' + name_doc
        elif isinstance(v_type, TensorType):
            dtype_doc = self(v_type.dtype)
            name_doc = self(v)
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
            name_doc = self(v)
            if v_type.use_bracket:
                return attr_doc + base_type_doc + ' ' + name_doc + '[]'
            else:
                return attr_doc + base_type_doc + ' *' + name_doc
        elif isinstance(v_type, TensorPointerType):
            dtype_doc = self(v_type.tensor_type.dtype)
            name_doc = self(v)
            return dtype_doc + ' *' + name_doc
        else:
            assert False

    def __call__(self, node) -> Doc:
        return self.visit(node)

    def visit(self, node):
        # pylint: disable=too-many-return-statements
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
        elif isinstance(node, str):
            return Text(node)
        elif isinstance(node, Doc):
            return node
        else:
            raise ValueError(type(node))

    def visit_IRModule(self, module: IRModule) -> Doc:
        self.ir_module = module
        doc = Doc()
        # todo: only add necessary headers
        doc += Text('#include <stdint.h>') + NewLine()
        doc += Text('#include <cuda_fp16.h>') + NewLine()
        doc += Text('#include <cuda_bf16.h>') + NewLine()
        doc += Text('#include <hidet/runtime/cuda_context.h>') + NewLine()
        doc += Text('#include <hidet/runtime/cpu_context.h>') + NewLine()

        # nvcc use float to 'store' tfloat32 data
        doc += Text('typedef float tfloat32_t;') + NewLine()

        # According to here: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-altfp
        # there should be a function called '__float_to_tf32' in cuda c to convert float to tfloat32,
        # but I did not find such a function. By looking at cutlass's implementation of converting
        # float to tfloat32, it seems that we do not need to do anything to convert. Put a definition
        # here in case nvidia add the definition in the future.
        doc += Text('#define __float_to_tf32(x) (x)') + NewLine()

        if module.task is not None:
            doc += '/*' + NewLine()
            doc += str(module.task) + NewLine()
            doc += '*/' + NewLine()
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
        if func.kind == 'cuda_kernel':
            doc += '__global__'
        elif func.kind == 'cuda_device':
            doc += '__device__ __forceinline__'
        elif func.kind in ['packed_func', 'host_kernel']:
            doc += '__host__'

        doc += ' ' + self(func.ret_type)
        # doc += ' void'

        # launch bound for grid worker
        if func.kind == 'cuda_kernel':
            block_dim = func.attrs['cuda_block_dim']
            if isinstance(block_dim, list):
                block_dim = prod(block_dim)
            if 'cuda_min_blocks' in func.attrs:
                min_blocks = func.attrs['cuda_min_blocks']
                doc += f' __launch_bounds__({block_dim}, {min_blocks})'
            else:
                doc += f' __launch_bounds__({block_dim})'

        # func name
        canonized_func_name = self.canonize_funcname(func.name)
        doc += ' ' + canonized_func_name
        self.func_name_map[func.name] = canonized_func_name

        # parameters
        doc += '('
        param_docs = []
        for param in func.params:
            param_docs.append(self.param_declare(param))
        doc += doc_join(param_docs, Text(', '))
        doc += ') {'

        # comments
        label = func.get_attr('label', default=None, allow_missing=True)
        if label:
            doc += (NewLine() + '// label: {}'.format(label)).indent()

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
        return '(~' + self(e.base) + ')'

    def visit_LeftShift(self, e: LeftShift):
        return '(' + self(e.base) + ' << ' + self(e.cnt) + ')'

    def visit_RightShift(self, e: RightShift):
        return '(' + self(e.base) + ' >> ' + self(e.cnt) + ')'

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
                return '(int64_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.uint64:
                return '(uint64_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.int32:
                return '(int32_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.uint32:
                return '(uint32_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.int16:
                return '(int16_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.uint16:
                return '(uint16_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.int8:
                return '(int8_t)(' + self(e.expr) + ')'
            elif dst_dtype == dtypes.uint8:
                return '(uint8_t)(' + self(e.expr) + ')'
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
        raise NotImplementedError()

    def visit_Dereference(self, e: Dereference):
        return Text('*') + self(e.expr)

    def visit_Call(self, e: Call):
        func_name = e.func_var.hint
        if func_name in self.ir_module.functions:
            func = self.ir_module.lookup(func_name)
            func_name = Text(self.canonize_funcname(func_name))
            if func.kind == 'cuda_kernel':

                if isinstance(func.attrs['cuda_block_dim'], int) and func.attrs['cuda_block_dim'] > 1024:
                    raise ValueError('CUDA block dimension cannot be larger than 1024.')

                def dim3_str(dims):
                    if isinstance(dims, (int, Expr)):
                        return self(dims)
                    else:
                        return Text('dim3(') + self(dims) + ')'

                configs = [
                    dim3_str(func.attrs['cuda_grid_dim']),  # grid dimension
                    dim3_str(func.attrs['cuda_block_dim']),  # block dimension
                    func.attrs.get('cuda_dynamic_smem_bytes', 0),  # dynamic shared memory size
                    # cuda stream (get_cuda_stream() function is defined in hidet/runtime.h)
                    Text('(cudaStream_t)get_cuda_stream()'),
                ]
                launch_config = Text('<<<') + doc_join([self(v) for v in configs], sep=', ') + Text('>>>')
            else:
                launch_config = []
            param_doc = Text('(') + doc_join([self(arg) for arg in e.args], Text(', ')) + ')'
            return func_name + launch_config + param_doc
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
            msg = "Callee {} not found in current ir module, and it is not primitive function.".format(func_name)
            raise ValueError(msg)

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

    @staticmethod
    def scalar_literal(value, dtype: DataType):
        if dtype == dtypes.boolean:
            ret = 'true' if value else 'false'
        elif dtype == dtypes.float64:
            ret = '{}'.format(float(value))
        elif dtype == dtypes.float32:
            ret = '{}f'.format(float(value))
        elif dtype == dtypes.float16:
            ret = 'half({}f)'.format(float(value))
        elif dtype == dtypes.tfloat32:
            ret = '__float_to_tf32({}f)'.format(float(value))
        elif dtype == dtypes.bfloat16:
            ret = '__float2bfloat16({}f)'.format(float(value))
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
        else:
            raise NotImplementedError('Cannot recognize scalar literal {} with dtype {}'.format(value, dtype))
        return Text(ret)

    def visit_Constant(self, e: Constant):
        if e.is_scalar():
            return self.scalar_literal(e.value, e.type)
        else:
            assert isinstance(e.type, TensorType)
            dtype = e.type.dtype
            items = [self.scalar_literal(v, dtype) for v in np.array(e.value).flatten()]
            return '{' + doc_join(items, ', ') + '}'

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
        if stmt.unroll is not None:
            if isinstance(stmt.unroll, bool):
                if stmt.unroll:
                    doc += NewLine() + '#pragma unroll'  # complete unroll
                else:
                    doc += NewLine() + '#pragma unroll 1'  # prevent from unrolling
            else:
                assert isinstance(stmt.unroll, int)
                doc += NewLine() + '#pragma unroll {}'.format(stmt.unroll)
        doc += NewLine() + Text('for (') + init_doc + '; ' + cond_doc + '; ' + update_doc + ') '
        body_doc = self(stmt.body)
        doc += Text('{') + body_doc.indent() + NewLine() + Text('} ')
        return doc

    def visit_ForTaskStmt(self, stmt: ForTaskStmt):
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
            return NewLine() + Text('assert(((void)"') + stmt.msg + '", ' + self(stmt.cond) + '));'
        else:
            return NewLine() + Text('assert(') + self(stmt.cond) + ');'

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
            self.canonize_funcname(stmt.func_var.hint),
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

    def visit_ScalarType(self, t: DataType):
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
            'bfloat16': 'nv_bfloat16',
            'tfloat32': 'tfloat32_t',
        }
        return Text(scalar_type_map[t.name])

    def visit_TensorType(self, t: TensorType):
        return Text('TensorType(') + self(t.dtype) + ', [' + doc_join([self(s) for s in t.shape], ", ") + '])'

    def visit_PointerType(self, t: PointerType):
        return self(t.base_type) + Text('*')

    def visit_TensorPointerType(self, t: TensorPointerType):
        return self(t.tensor_type.dtype) + Text('*')

    def visit_ReferenceType(self, t: ReferenceType):
        raise ValueError()

    def visit_VoidType(self, t: VoidType):
        return Text('void')

    # the following expressions should not remain to codegen
    def visit_TensorSlice(self, e: TensorSlice):
        raise ValueError()

    def visit_ScalarNode(self, e: ScalarNode):
        raise ValueError()

    def visit_TensorNode(self, e: TensorNode):
        raise ValueError()

    def visit_AnyExpr(self, e: AnyExpr):
        raise ValueError()


def codegen(ir_module: IRModule, src_out_path: Optional[str] = None) -> str:
    gen = Codegen()
    doc = gen(ir_module)
    code = str(doc)
    if src_out_path is not None:
        dir_path = os.path.dirname(src_out_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(src_out_path, 'w') as f:
            f.write(code)
    return code
