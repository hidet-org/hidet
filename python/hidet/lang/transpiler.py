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
# pylint: disable=import-outside-toplevel, too-many-branches, too-many-locals
from __future__ import annotations
import types
import math
import builtins
import operator

from typing import Optional, Dict, Any, Union, Type, Callable
import os.path
import ast
from ast import Module

# statements
from ast import FunctionDef, Return, Assign, AnnAssign, AugAssign, For, While, If, With, Assert, Expr, Pass, Break
from ast import Continue, Nonlocal, Global

# expressions
from ast import Constant, Num, Str, NameConstant
from ast import BoolOp, BinOp, UnaryOp, Lambda, IfExp, Compare, Call, Attribute, Subscript, Starred, Name, Tuple, Slice
from ast import In, NotIn
from ast import ExtSlice, List
from ast import ListComp, DictComp, SetComp, GeneratorExp, comprehension

# expr context
from ast import Load, Store, Del

# arithmetic and bitwise operators
from ast import UAdd, USub, Add, Sub, Mult, Div, FloorDiv, Mod, Pow, BitOr, BitXor, BitAnd, Invert, LShift, RShift

# bool and compare operators
from ast import Not, And, Or, Eq, NotEq, Lt, LtE, Gt, GtE

from ast import Index

import astunparse
from hidet import ir
from hidet.ir.expr import Var
from hidet.ir.stmt import DeclareScope
from hidet.ir.tools import simplify
from hidet.ir.builders import FunctionBuilder
from hidet.utils import red, bold, blue, str_indent
import hidet.lang.attrs
from hidet.lang.constructs.loops import HidetLoopIterable
from hidet.lang.constructs.declare import Declaration
from hidet.lang.constructs.meta import HidetMetaLoopIterable


class HidetProgramError(Exception):
    def __init__(self, translator: PythonAstFunctor, obj, msg: str):
        super().__init__(translator, obj, msg)  # make this exception picklable
        assert isinstance(obj, ast.AST)
        self.file = translator.file
        self.lineno = translator.start_lineno + obj.lineno
        self.column = translator.start_column + obj.col_offset
        self.msg = msg

    def __str__(self):
        lines = []
        if not os.path.exists(self.file):
            source_line = ''
        else:
            with open(self.file, 'r') as f:
                source_lines = list(f.readlines())
                if self.lineno < len(source_lines):
                    source_line = source_lines[self.lineno - 2].rstrip()
                else:
                    source_line = ''
        lines.append('')
        lines.append(
            '  File {file}:{line}:{column}:'.format(
                file=os.path.abspath(self.file), line=self.lineno - 1, column=self.column
            )
        )
        if source_line:
            lines.append(source_line)
            lines.append(' ' * self.column + bold(red('^')))
        if source_line and '\n' not in self.msg:
            indent = self.column
        else:
            indent = 4
        lines.append('{msg}'.format(msg=blue(str_indent(self.msg, indent=indent))))
        return '\n'.join(lines)


class PythonAstFunctor:
    """
    A base functor for Python AST nodes.

    Please refers to https://docs.python.org/3/library/ast.html for more details about Python AST.
    """

    def __init__(self, file: str, start_lineno: int, start_column: int):
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column

    def __call__(self, node):
        return self.visit(node)

    def visit(self, node):
        from hidet.ir.library.tune import ScheduleError

        method = 'visit_' + node.__class__.__name__
        if hasattr(self, method):
            visitor = getattr(self, method)
        else:
            msg = 'The AST node {} does not support in HidetScript.'.format(node.__class__.__name__)
            raise HidetProgramError(self, node, msg)

        try:
            return visitor(node)
        except ScheduleError:
            raise
        except HidetProgramError:
            raise
        except Exception as e:
            # import traceback
            raise HidetProgramError(self, node, 'Internal exception occurred during transpiling this ast node.') from e

    def visit_Module(self, module: Module):
        raise NotImplementedError()

    def visit_FunctionDef(self, func_def: FunctionDef):
        raise NotImplementedError()

    def visit_Return(self, stmt: Return):
        raise NotImplementedError()

    def visit_Assign(self, stmt: Assign):
        raise NotImplementedError()

    def visit_AnnAssign(self, stmt: AnnAssign):
        raise NotImplementedError()

    def visit_AugAssign(self, stmt: AugAssign):
        raise NotImplementedError()

    def visit_For(self, stmt: For):
        raise NotImplementedError()

    def visit_While(self, stmt: While):
        raise NotImplementedError()

    def visit_If(self, stmt: If):
        raise NotImplementedError()

    def visit_With(self, stmt: With):
        raise NotImplementedError()

    def visit_Assert(self, stmt: Assert):
        raise NotImplementedError()

    def visit_Expr(self, stmt: Expr):
        raise NotImplementedError()

    def visit_Pass(self, stmt: Pass):
        raise NotImplementedError()

    def visit_Break(self, stmt: Break):
        raise NotImplementedError()

    def visit_Continue(self, stmt: Continue):
        raise NotImplementedError()

    def visit_BoolOp(self, expr: BoolOp):
        raise NotImplementedError()

    def visit_BinOp(self, expr: BinOp):
        raise NotImplementedError()

    def visit_UnaryOp(self, expr: UnaryOp):
        raise NotImplementedError()

    def visit_Lambda(self, expr: Lambda):
        raise NotImplementedError()

    def visit_IfExp(self, expr: IfExp):
        raise NotImplementedError()

    def visit_Compare(self, expr: Compare):
        raise NotImplementedError()

    def visit_Call(self, expr: Call):
        raise NotImplementedError()

    def visit_Constant(self, expr: Constant):
        raise NotImplementedError()

    def visit_Num(self, expr: Num):
        return self.visit(ast.copy_location(Constant(expr.n), expr))

    def visit_Str(self, expr: Str):
        return self.visit(ast.copy_location(Constant(expr.s), expr))

    def visit_NameConstant(self, expr: NameConstant):
        return self.visit(ast.copy_location(Constant(expr.value), expr))

    def visit_Attribute(self, expr: Attribute):
        raise NotImplementedError()

    def visit_Subscript(self, expr: Subscript):
        raise NotImplementedError()

    def visit_Starred(self, expr: Starred):
        raise NotImplementedError(astunparse.unparse(expr))

    def visit_Name(self, expr: Name):
        raise NotImplementedError()

    def visit_Tuple(self, expr: Tuple):
        raise NotImplementedError()

    def visit_List(self, expr: List):
        raise NotImplementedError()

    def visit_Slice(self, expr: Slice):
        raise NotImplementedError()

    def visit_ExtSlice(self, expr: ExtSlice):
        raise NotImplementedError()

    def visit_Index(self, expr: Index):
        raise NotImplementedError()

    def visit_ListComp(self, expr: ListComp):
        raise NotImplementedError()

    def visit_SetComp(self, expr: SetComp):
        raise NotImplementedError()

    def visit_DictComp(self, expr: DictComp):
        raise NotImplementedError()

    def visit_GeneratorExp(self, expr: GeneratorExp):
        raise NotImplementedError()

    def visit_Nonlocal(self, stmt: Nonlocal):
        raise NotImplementedError()


HostTypes = (ir.TaskMapping, ir.DataLayout, float, int, type(None))


class Scope:
    def __init__(self, parent: Optional[Scope]):
        self.parent: Optional[Scope] = parent
        self.name2var: Dict[str, Var] = {}
        self.name2host_var: Dict[str, Any] = {}
        self.stmts: list[ir.Stmt] = []
        self.attributes: dict[str, Any] = {}

    @staticmethod
    def default_top_level():
        scope = Scope(None)
        scope.define_host_var('range', hidet.lang.constructs.loops.range)
        return scope

    def define_var(self, name: str, v: Var):
        if name == '_':
            # ignore anonymous variable '_'
            return
        self.name2var[name] = v

    def define_host_var(self, name: str, value: Any):
        self.name2host_var[name] = value

    def lookup(self, name: str, search_parents=True) -> Optional[Union[Var, HostTypes]]:
        if name in self.name2var:
            return self.name2var[name]
        if name in self.name2host_var:
            return self.name2host_var[name]
        if search_parents and self.parent:
            return self.parent.lookup(name, search_parents)
        return None

    def annotate(self, name: str, value: Any):
        if name in self.attributes:
            raise ValueError('Attribute {} has already been annotated.'.format(name))
        self.attributes[name] = value

    def append(self, stmt: ir.Stmt):
        self.stmts.append(stmt)

    def flush_stmts(self) -> ir.Stmt:
        seq_stmt = ir.SeqStmt(seq=list(self.stmts))
        self.stmts.clear()
        return seq_stmt


class ScopeStack:
    def __init__(self):
        self.scopes: list[Scope] = [Scope.default_top_level()]

    def __enter__(self) -> Scope:
        parent = self.scopes[-1]
        scope = Scope(parent)
        self.scopes.append(scope)
        return scope

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()


class PythonToHidetTranslator(PythonAstFunctor):
    def __init__(self, file, start_lineno, start_column, env, func_annotations):
        super().__init__(file, start_lineno, start_column)
        self.env: Dict[str, Any] = env
        self.func_annotations: Dict[str, Any] = func_annotations

        self.fb: Optional[FunctionBuilder] = None
        self.scope_stack: ScopeStack = ScopeStack()

    def scope(self):
        return self.scope_stack

    @property
    def current_scope(self) -> Scope:
        if len(self.scope_stack.scopes) == 0:
            raise ValueError('The scope stack is empty.')
        return self.scope_stack.scopes[-1]

    def process_assign(self, lhs: Union[Attribute, Subscript, Name], rhs, type_annotation: Optional[ast.expr] = None):
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        # check the rhs value, must be an instance of allowed_types or a list of these kinds of elements.
        host_var_types = (ir.TaskMapping, ir.DataLayout, ir.TensorSlice, ir.Function, str, list, tuple, dict)
        allowed_types = (ir.Expr, ir.BaseType, Declaration, float, int, str, type(None))
        allowed_types += host_var_types
        assert isinstance(rhs, allowed_types) or (
            isinstance(rhs, list) and all(isinstance(v, allowed_types) for v in rhs)
        ), 'unexpected value "{}" with type {}'.format(rhs, type(rhs))

        # three cases of assignment:
        #    1. v = ...
        #    2. a[i, j] = ...
        #    3. attr.name = ...
        if isinstance(lhs, Name):
            var_name = lhs.id
            lookup_result = self.current_scope.lookup(var_name, search_parents=True)
            if type_annotation is not None or lookup_result is None or not isinstance(lookup_result, Var):
                # There are two cases of variable definition:
                #   1) assignment with type annotation, or
                #   2) the used name has not been defined yet.
                if type_annotation is not None:
                    if self.current_scope.lookup(var_name, search_parents=False) is not None:
                        msg = 'Can not define two variables with the same name in the same scope.'
                        raise HidetProgramError(self, lhs, msg)
                    var_type = self.visit(type_annotation)
                    var = Var(hint=var_name, type=var_type)
                    self.current_scope.define_var(name=var_name, v=var)
                    self.current_scope.append(ir.DeclareStmt(var, init=rhs))
                else:
                    # define a new variable
                    # during transpiling, there are two kinds of variables:
                    #   1. host variable, the variable in host language, and
                    #   2. hidet variable, the variable in hidet.
                    # Typical host variables are like TaskMapping, DataLayout that are not scalar or tensor.
                    # We use host variable to reduce the complexity of hidet's data model.
                    if isinstance(rhs, host_var_types):
                        self.current_scope.define_host_var(var_name, rhs)
                    else:
                        init_value = None
                        is_static = False
                        scope = DeclareScope.Default
                        if isinstance(rhs, ir.BaseType):
                            var_type = rhs
                        elif isinstance(rhs, Declaration):
                            var_type = rhs.type
                            is_static = rhs.is_static
                            scope = rhs.scope
                            init_value = rhs.init
                        else:
                            rhs = ir.convert(rhs)
                            var_type = ir.infer_type(rhs)
                            init_value = rhs
                        var = Var(hint=var_name, type=var_type)
                        self.current_scope.append(
                            ir.DeclareStmt(var, init=init_value, is_static=is_static, scope=scope)
                        )
                        self.current_scope.define_var(name=var_name, v=var)
            else:
                # In other cases, it is an assignment of defined variable.
                var = lookup_result
                self.current_scope.append(ir.AssignStmt(var, value=rhs))
        elif isinstance(lhs, Subscript):
            # example: a[3, 4] = 5.0
            base = self.visit(lhs.value)
            indices = self.visit(lhs.slice)
            if not isinstance(indices, list):
                indices = [indices]
            self.current_scope.append(ir.BufferStoreStmt(buf=base, indices=indices, value=rhs))
        elif isinstance(lhs, Attribute):
            # example: attr.cuda.block_dim = 16, 16
            lhs_base = self.visit(lhs.value)
            namespace = {hidet.lang.attrs: '', hidet.lang.attrs.cuda: 'cuda.'}
            if lhs_base in namespace:
                attr_name = namespace[lhs_base] + lhs.attr
                if attr_name in ['cuda.block_dim', 'cuda.grid_dim', 'cuda.dynamic_smem_bytes']:
                    if isinstance(rhs, (tuple, list)):
                        rhs = [simplify(v) for v in rhs]
                    else:
                        rhs = simplify(rhs)
                self.current_scope.annotate(attr_name, rhs)
            else:
                raise HidetProgramError(self, lhs, 'Invalid assignment.')
        else:
            type_name = type(lhs).__name__
            raise HidetProgramError(self, lhs, 'Cannot recognize "{}" as left side of assignment.'.format(type_name))

    def visit_Module(self, module: Module):
        if len(module.body) != 1 or not isinstance(module.body[0], FunctionDef):
            msg = 'The module expects to have only one function definition statement, got\n'
            msg += str(astunparse.unparse(module))
            raise ValueError(msg)
        return self.visit(module.body[0])

    def _process_arg_type(self, arg, arg_type: Union[ir.BaseType, Declaration, Type[int], Type[float], Type[bool]]):
        if isinstance(arg_type, ir.BaseType):
            if isinstance(arg_type, ir.TensorType):
                # we automatically change the tensor type of argument to a tensor pointer type.
                arg_type = ir.tensor_pointer_type(dtype=arg_type.dtype, shape=arg_type.shape, layout=arg_type.layout)
        elif isinstance(arg_type, Declaration):
            arg_type = arg_type.type
            if isinstance(arg_type, ir.TensorType):
                # we automatically change the tensor type of argument to a tensor pointer type.
                arg_type = ir.tensor_pointer_type(dtype=arg_type.dtype, shape=arg_type.shape, layout=arg_type.layout)
        elif arg_type in [bool, int, float]:
            type_dict = {bool: ir.data_type('bool'), int: ir.data_type('int32'), float: ir.data_type('float32')}
            arg_type = type_dict[arg_type]
        elif isinstance(arg_type, str):
            raise HidetProgramError(
                self,
                arg,
                (
                    'A python string as parameter type annotation detected. \n'
                    'This is usually because "from __future__ import annotations" has been used.\n'
                    'Currently, hidet script is not compatible with this feature. \n'
                    'Please considering not using it in module that defines hidet script.'
                ),
            )
        else:
            raise HidetProgramError(self, arg, 'Hidet expect a type annotation for this parameter.')
        return arg_type

    def visit_FunctionDef(self, func_def: FunctionDef):
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements, import-outside-toplevel
        func_params = []
        with self.scope() as env_scope:
            for name, value in self.env.items():
                env_scope.define_host_var(name, value)
            with self.scope() as scope:
                # process function arguments
                args: ast.arguments = func_def.args
                if args.vararg is not None:
                    raise HidetProgramError(self, args.vararg, 'Hidet program does not support "*args" arguments.')
                if len(args.kwonlyargs) != 0:
                    raise HidetProgramError(
                        self, args.kwonlyargs[0], 'Hidet program does not support "*kwargs" arguments.'
                    )
                if args.kwarg is not None:
                    raise HidetProgramError(self, args.kwarg, 'Hidet program does not support keyword arguments.')
                if len(args.kw_defaults) > 0:
                    raise HidetProgramError(self, args.kw_defaults[0], 'Hidet does not support default argument.')
                if len(args.defaults) > 0:
                    raise HidetProgramError(self, args.defaults[0], 'Hidet does not support default argument.')
                for arg in args.args:
                    from hidet.lang.constructs.meta import HidetMetaParamTypeList

                    arg_name = arg.arg
                    if arg_name not in self.func_annotations:
                        raise HidetProgramError(self, arg, 'Hidet expects type annotation for each function argument.')
                    arg_type = self.func_annotations[arg_name]

                    if isinstance(arg_type, HidetMetaParamTypeList):
                        arg_types = [self._process_arg_type(arg, t) for t in arg_type.arg_types]
                        param_vars = [Var(hint=arg_name, type=t) for t in arg_types]
                        func_params.extend(param_vars)
                        scope.define_host_var(arg_name, list(param_vars))
                    else:
                        arg_type: ir.BaseType = self._process_arg_type(arg, arg_type)
                        param_var = Var(hint=arg_name, type=arg_type)
                        func_params.append(param_var)
                        scope.define_var(arg_name, param_var)

                # process function body
                for stmt in func_def.body:
                    self.visit(stmt)

            # return type
            if func_def.returns is None:
                # the default return type is void
                ret_type = ir.VoidType()
            else:
                # ret_type = self.visit(func_def.returns)
                ret_type = self.func_annotations['return']
                if not isinstance(ret_type, ir.BaseType):
                    if ret_type is bool:
                        ret_type = ir.data_type('bool')
                    elif ret_type is int:
                        ret_type = ir.data_type('int32')
                    elif ret_type is float:
                        ret_type = ir.data_type('float32')
                    else:
                        raise HidetProgramError(self, func_def.returns, 'Expect a type of function return value.')

            # get function attributes
            func_attrs: Dict[str, Any] = scope.attributes.copy()
            if 'func_kind' in func_attrs:
                func_kind = func_attrs['func_kind']
            elif 'cuda.grid_dim' in func_attrs or 'cuda.block_dim' in func_attrs:
                if not all(name in func_attrs for name in ['cuda.grid_dim', 'cuda.block_dim']):
                    raise HidetProgramError(
                        self, func_def, 'CUDA kernel expects to have both attrs.cuda.grid_dim and attrs.cuda.block_dim.'
                    )
                func_kind = 'cuda_kernel'
            else:
                func_kind = 'cuda_internal'
            func_name = func_attrs.get('func_name', func_def.name)

        return ir.Function(
            name=func_name,
            params=func_params,
            body=scope.flush_stmts(),
            ret_type=ret_type,
            kind=func_kind,
            attrs=func_attrs,
        )

    def visit_Assign(self, stmt: Assign):
        # pylint: disable=too-many-branches
        if len(stmt.targets) > 1:
            raise HidetProgramError(self, stmt, 'Hidet does not support syntax like "a = b = 1".')
        target = stmt.targets[0]
        value = stmt.value
        if isinstance(target, (Tuple, List)):
            lhs_list = target.elts
        else:
            lhs_list = [target]

        if isinstance(value, (Tuple, List)):
            rhs_list = [self.visit(v) for v in value.elts]
        else:
            rhs_list = [self.visit(value)]

        if len(lhs_list) == len(rhs_list):
            for lhs, rhs in zip(lhs_list, rhs_list):
                self.process_assign(lhs, rhs)
        elif len(lhs_list) == 1:
            lhs = lhs_list[0]
            assert isinstance(lhs, (Subscript, Name, Attribute))
            self.process_assign(lhs, rhs_list)
        elif len(rhs_list) == 1:
            if isinstance(rhs_list[0], ir.Expr):
                raise HidetProgramError(self, stmt, 'Hidet does not support unpacking.')
            rhs_list = list(rhs_list[0])
            if len(lhs_list) != len(rhs_list):
                raise HidetProgramError(
                    self, stmt, 'Trying to assign {} values to {} objects'.format(len(rhs_list), len(lhs_list))
                )
            for lhs, rhs in zip(lhs_list, rhs_list):
                self.process_assign(lhs, rhs)
        else:
            raise HidetProgramError(
                self, stmt, 'Can not assign {} elements to {} elements.'.format(len(rhs_list), len(lhs_list))
            )

    def visit_Name(self, expr: Name):
        if isinstance(expr.ctx, Store):
            raise ValueError('Internal Error, please deal with all Store behavior in parent nodes like Assign.')
        elif isinstance(expr.ctx, Load):
            name: str = expr.id
            var: Optional[Var] = self.current_scope.lookup(name)
            if var is None:
                if name in self.env:
                    # access external variable, such as thread_x
                    return self.env[name]
                if name in builtins.__dict__:
                    # access builtin functions such as max, min
                    return builtins.__dict__[name]
                raise HidetProgramError(self, expr, 'Trying to access variable without definition.')
            return var
        elif isinstance(expr.ctx, Del):
            raise HidetProgramError(self, expr, 'Hidet does not support del statement.')
        else:
            raise ValueError()

    def visit_Tuple(self, expr: Tuple):
        return [self.visit(v) for v in expr.elts]

    def visit_List(self, expr: List):
        return [self.visit(v) for v in expr.elts]

    def visit_BinOp(self, expr: BinOp):
        lhs = self.visit(expr.left)
        rhs = self.visit(expr.right)
        if isinstance(lhs, ir.DataLayout) and isinstance(rhs, ir.DataLayout):
            if isinstance(expr.op, Mult):
                return lhs * rhs
            elif isinstance(expr.op, Add):
                return lhs + rhs
            else:
                raise HidetProgramError(self, expr, 'Hidet does not support this operation on DataLayout.')
        elif isinstance(lhs, ir.TaskMapping) and isinstance(rhs, ir.TaskMapping):
            assert isinstance(expr.op, Mult)
            return lhs * rhs
        elif isinstance(lhs, str) and isinstance(rhs, str):
            assert isinstance(expr.op, Add)
            return lhs + rhs
        elif isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
            assert isinstance(expr.op, Add)
            return list(lhs) + list(rhs)
        elif isinstance(lhs, (ir.Expr, float, int)) and isinstance(rhs, (ir.Expr, float, int)):
            # pylint: disable=import-outside-toplevel
            from hidet.ir import primitives

            op_dict = {
                Add: operator.add,
                Sub: operator.sub,
                Mult: operator.mul,
                Div: operator.truediv,
                FloorDiv: operator.floordiv,
                Mod: operator.mod,
                BitXor: operator.xor,
                BitOr: operator.or_,
                BitAnd: operator.and_,
                Pow: primitives.pow,
                LShift: ir.expr.left_shift,
                RShift: ir.expr.right_shift,
            }

            if type(expr.op) in op_dict:
                return op_dict[type(expr.op)](lhs, rhs)
            else:
                type_name = type(expr.op).__name__
                raise HidetProgramError(self, expr, 'Currently, we do not support {} operator.'.format(type_name))
        else:
            raise HidetProgramError(
                self, expr, 'Can not apply operator {} to {} and {}.'.format(expr.op, type(lhs), type(rhs))
            )

    def visit_BoolOp(self, expr: BoolOp):
        values = [self.visit(v) for v in expr.values]
        if isinstance(expr.op, And):
            return ir.logical_and(*values)
        else:
            assert isinstance(expr.op, Or)
            return ir.logical_or(*values)

    def visit_Compare(self, expr: Compare):
        front = self.visit(expr.left)
        op_dict = {
            And: ir.logical_and,
            Or: ir.logical_or,
            Eq: ir.equal,
            Gt: lambda a, b: ir.less_than(b, a),  # pylint: disable=arguments-out-of-order
            Lt: ir.less_than,
            GtE: lambda a, b: ir.less_equal(b, a),  # pylint: disable=arguments-out-of-order
            LtE: ir.less_equal,
            NotEq: ir.not_equal,
        }
        py_op_dict = {
            And: operator.and_,
            Or: operator.or_,
            Eq: operator.eq,
            Gt: operator.gt,
            Lt: operator.lt,
            GtE: operator.ge,
            LtE: operator.le,
            NotEq: operator.ne,
            In: lambda a, b: a in b,
            NotIn: lambda a, b: a not in b,
        }
        cond = None
        comparators = [self.visit(v) for v in expr.comparators]
        for op, current in zip(expr.ops, comparators):
            op_kind = type(op)
            if isinstance(front, ir.Node) or isinstance(current, ir.Node):
                if op_kind not in op_dict:
                    raise HidetProgramError(
                        self, expr, 'Currently, we do not support {} operator for hidet vars.'.format(op_kind.__name__)
                    )
                cur_cond = op_dict[op_kind](front, current)
            else:
                cur_cond = py_op_dict[op_kind](front, current)
            cond = ir.logical_and(cond, cur_cond) if cond is not None else cur_cond
            front = current
        return cond

    def visit_UnaryOp(self, expr: UnaryOp):
        value = self.visit(expr.operand)
        if isinstance(value, hidet.ir.Node):
            if isinstance(expr.op, Not):
                # not v
                assert isinstance(value, ir.Expr)
                return ir.logical_not(value)
            elif isinstance(expr.op, Invert):
                # there are two cases for a ~ operator: ~something
                # case 1: get the address of an expression
                # case 2: get the pointer type that points to the given type
                from hidet.ir.expr import Address
                from hidet.ir.type import BaseType

                if isinstance(value, BaseType):
                    return ~value
                else:
                    assert isinstance(value, ir.Expr)
                    return Address(value)
            elif isinstance(expr.op, UAdd):
                # +v
                return value
            elif isinstance(expr.op, USub):
                # -v
                assert isinstance(value, ir.Expr)
                return -value
            else:
                raise HidetProgramError(self, expr, 'Can not recognize unary operator.')
        else:
            op_dict: Dict[Type[Union[UAdd, USub, Not].unaryop], Callable] = {
                UAdd: operator.pos,
                USub: operator.neg,
                Not: operator.not_,
            }
            return op_dict[type(expr.op)](value)

    def visit_If(self, stmt: If):
        cond = self.visit(stmt.test)

        if isinstance(cond, ir.Constant):
            cond = bool(cond)

        if isinstance(cond, bool):
            if cond:
                for s in stmt.body:
                    self.visit(s)
            else:
                for s in stmt.orelse:
                    self.visit(s)
        else:
            with self.scope() as then_scope:
                for s in stmt.body:
                    self.visit(s)

            with self.scope() as else_scope:
                for s in stmt.orelse:
                    self.visit(s)

            then_body = then_scope.flush_stmts()
            else_body = else_scope.flush_stmts() if len(stmt.orelse) > 0 else None
            self.current_scope.append(ir.IfStmt(cond=cond, then_body=then_body, else_body=else_body))

    def visit_Index(self, expr: Index):
        return self.visit(expr.value)

    def visit_Constant(self, expr: Constant):
        if isinstance(expr.value, (float, int)):
            return expr.value
        elif isinstance(expr.value, str):
            return expr.value
        elif expr.value is None:
            return expr.value
        else:
            raise HidetProgramError(self, expr, 'Can not recognize Constant {}'.format(repr(expr.value)))

    def visit_For(self, stmt: For):
        # create loop vars
        iter_targets: list[Name] = []
        if isinstance(stmt.target, (List, Tuple)):
            for target in stmt.target.elts:
                if not isinstance(target, Name):
                    raise HidetProgramError(self, stmt, 'For loop target must be a name.')
                iter_targets.append(target)
        else:
            if not isinstance(stmt.target, Name):
                raise HidetProgramError(self, stmt, 'For loop target must be a name.')
            iter_targets.append(stmt.target)

        # construct for body
        stmt_iter = self.visit(stmt.iter)
        num_targets: int = len(iter_targets)
        if isinstance(stmt_iter, HidetLoopIterable):
            loop_vars: list[Var] = []
            host_vars: Dict[str, Any] = {}

            num_loop_vars: int = stmt_iter.num_loop_vars()

            if num_targets == num_loop_vars > 1 or (num_targets == num_loop_vars == 1 and not stmt_iter.bind_tuple()):
                for target in iter_targets:
                    loop_vars.append(Var(target.id, type=ir.data_type('int32')))
            elif num_targets == 1:
                name = iter_targets[0].id
                for i in range(num_loop_vars):
                    loop_vars.append(Var(f'{name}{i}', type=ir.data_type('int32')))
                host_vars[name] = list(loop_vars)
            else:
                raise HidetProgramError(
                    self, stmt, f'Expect {num_loop_vars} loop variables, but got {len(iter_targets)}.'
                )

            with self.scope() as for_scope:
                for var in loop_vars:
                    for_scope.define_var(name=var.hint, v=var)
                for name, value in host_vars.items():
                    for_scope.define_host_var(name, value)
                for s in stmt.body:
                    self.visit(s)
            body = for_scope.flush_stmts()
            for_stmt = stmt_iter.generate_loop_statement(loop_vars=loop_vars, body=body)
            self.current_scope.append(for_stmt)
        elif isinstance(stmt_iter, HidetMetaLoopIterable):
            for host_value in stmt_iter:
                if num_targets == 1:
                    self.current_scope.define_host_var(iter_targets[0].id, host_value)
                else:
                    unpacked_host_value = list(host_value)
                    if len(unpacked_host_value) != num_targets:
                        raise HidetProgramError(
                            self,
                            stmt.target,
                            f'Can not unpack {len(unpacked_host_value)} values to {num_targets} targets.',
                        )
                    for name, value in zip(iter_targets, unpacked_host_value):
                        self.current_scope.define_host_var(name.id, value)
                for s in stmt.body:
                    self.visit(s)
        else:
            msg = (
                'For loop iterable must be a one of the following types: \n'
                '1.\n'
                '  for ... in range(...): \n'
                '      ...\n'
                '2.\n'
                '  for ... in grid(...): \n'
                '      ...\n'
                '3.\n'
                '  for ... in task_mapping.on(...): \n'
                '      ...'
            )
            raise HidetProgramError(self, stmt.iter, msg)

    def visit_AugAssign(self, stmt: AugAssign):
        if isinstance(stmt.target, Name):
            target = Name(stmt.target.id, Load())
            var_value = self.visit(target)
        else:
            var_value = self.visit(stmt.target)
            # raise NotImplementedError()
        value = self.visit(stmt.value)
        op_dict = {
            Add: operator.add,
            Sub: operator.sub,
            Mult: operator.mul,
            Div: operator.truediv,
            FloorDiv: operator.floordiv,
            Mod: operator.mod,
        }
        result_value = op_dict[type(stmt.op)](var_value, value)
        assert isinstance(stmt.target, (Name, Subscript))
        self.process_assign(stmt.target, result_value)

    def visit_Subscript(self, expr: Subscript):
        base = self.visit(expr.value)
        indices = self.visit(expr.slice)
        return base[indices]

    def visit_Attribute(self, expr: Attribute):
        base = self.visit(expr.value)
        attr = expr.attr
        if hasattr(base, attr):
            return getattr(base, attr)
        else:
            raise HidetProgramError(self, expr, 'Can not access attribute "{}" of this object.'.format(attr))

    def visit_IfExp(self, expr: IfExp):
        cond = self.visit(expr.test)
        then_expr = self.visit(expr.body)
        else_expr = self.visit(expr.orelse)
        return ir.expr.if_then_else(cond, then_expr, else_expr)

    def visit_Expr(self, expr: Expr):
        value = self.visit(expr.value)
        if isinstance(value, ir.Call):
            self.current_scope.append(ir.EvaluateStmt(value))
        elif isinstance(value, ir.Stmt):
            # buf.write([i, j], value) would return a BufferStoreStmt
            self.current_scope.append(value)
        elif isinstance(value, str):
            # a """...""" string would be returned as a str
            # skip it
            pass
        else:
            raise HidetProgramError(self, expr, f'Can not recognize expression statement with type {type(value)}.')

    def visit_Call(self, expr: Call):
        func = self.visit(expr.func)
        args = []
        for arg in expr.args:
            if isinstance(arg, Starred):
                args.extend(self.visit(arg.value))
            else:
                args.append(self.visit(arg))

        if len(expr.keywords) == 0:
            kwargs = {}
        elif len(expr.keywords) == 1 and expr.keywords[0].arg is None:
            # func(a, b, **kwargs)
            kwargs = self.visit(expr.keywords[0].value)
        else:
            # func(a=1, b=2, c=3)
            kwargs = {kwarg.arg: self.visit(kwarg.value) for kwarg in expr.keywords}

        if isinstance(func, types.FunctionType):
            # call python function
            return func(*args, **kwargs)
        elif isinstance(func, types.MethodType):
            # call python class method
            return func(*args, **kwargs)
        elif isinstance(func, ir.Var):
            from hidet.ir.tools import infer_type

            # call hidet function
            if len(kwargs) > 0:
                raise HidetProgramError(self, expr, 'Hidet do not support call with keyword.')
            func_type: ir.FuncType = infer_type(func)
            assert isinstance(func_type, ir.FuncType)
            if len(func_type.param_types) != len(args):
                msg = 'The number of parameters of callee and given arguments does not match.'
                raise HidetProgramError(self, expr, msg)
            return func(*args)
        elif isinstance(func, ir.Function):
            # call a function defined as hidet script, like
            # @hidet.script
            # def f():
            #     ....
            from hidet.lang.script import ScriptModuleContext

            ctx = ScriptModuleContext.current_context()
            func_var = ctx.lookup(func.name)
            if func_var is None:
                raise HidetProgramError(self, expr, 'Call undefined function.')
            if len(kwargs) > 0:
                raise HidetProgramError(self, expr, 'Hidet do not support call with keyword.')
            assert isinstance(func_var.type, ir.FuncType)
            if len(func_var.type.param_types) != len(args):
                raise HidetProgramError(
                    self, expr, 'The number of parameters of callee and given arguments does not match.'
                )
            if func.kind == 'cuda_kernel':
                return ir.stmt.launch_kernel(
                    func_var=func_var,
                    args=args,
                    grid_dim=func.attrs['cuda.grid_dim'],
                    block_dim=func.attrs['cuda.block_dim'],
                    shared_mem=func.attrs.get('cuda.dynamic_smem_bytes', 0),
                )
            else:
                return func_var(*args)
        elif isinstance(func, (types.BuiltinMethodType, types.BuiltinFunctionType)):
            # call python builtin method, such "a string".format(...) or max, min
            from hidet.ir import primitives

            if all(not isinstance(arg, ir.Node) for arg in args):
                # pure python function call
                return func(*args, **kwargs)
            else:
                if any(not isinstance(arg, (ir.Expr, int, float, bool)) for arg in args):
                    # if any argument is not a valid expression
                    return func(*args, **kwargs)
                # overload hidet primitive, such as max, min
                func_map = {
                    builtins.max: (2, primitives.max),
                    builtins.min: (2, primitives.min),
                    math.exp: (1, primitives.exp),
                    math.log: (1, primitives.log),
                    math.sqrt: (1, primitives.sqrt),
                    math.sin: (1, primitives.sin),
                    math.cos: (1, primitives.cos),
                    math.tan: (1, primitives.tan),
                    math.asin: (1, primitives.asin),
                    math.acos: (1, primitives.acos),
                    math.atan: (1, primitives.atan),
                    math.sinh: (1, primitives.sinh),
                    math.cosh: (1, primitives.cosh),
                    math.tanh: (1, primitives.tanh),
                    math.asinh: (1, primitives.asinh),
                    math.acosh: (1, primitives.acosh),
                    math.atanh: (1, primitives.atanh),
                    math.ceil: (1, primitives.ceil),
                    math.floor: (1, primitives.floor),
                    math.trunc: (1, primitives.trunc),
                    math.isnan: (1, primitives.isnan),
                    math.isinf: (1, primitives.isinf),
                }
                if len(kwargs) > 0:
                    msg = 'Hidet do not support calling builtin function with keyword argument.'
                    raise HidetProgramError(self, expr, msg)
                if func in func_map:
                    arity, hidet_func = func_map[func]
                    if len(args) != arity:
                        msg = f'Hidet builtin function "{func.__name__}" takes {arity} arguments.'
                        raise HidetProgramError(self, expr, msg)
                    return hidet_func(*args)
                else:
                    raise HidetProgramError(
                        self,
                        expr,
                        'Currently, do not support calling python builtin function "{}".'.format(func.__qualname__),
                    )
        else:
            return func(*args, **kwargs)

    def visit_Starred(self, expr: Starred):
        raise HidetProgramError(self, expr, 'Hidet do not support unpack operator.')

    def visit_ExtSlice(self, expr: ExtSlice):
        return [self.visit(v) for v in expr.dims]

    def visit_Slice(self, expr: Slice):
        return slice(
            self.visit(expr.lower) if expr.lower is not None else None,
            self.visit(expr.upper) if expr.upper is not None else None,
            self.visit(expr.step) if expr.step is not None else None,
        )

    def visit_Assert(self, stmt: Assert):
        cond = self.visit(stmt.test)
        msg = None if stmt.msg is None else self.visit(stmt.msg)
        if stmt.msg is not None and not isinstance(msg, str):
            raise HidetProgramError(self, stmt.msg, 'Expect a string message.')
        self.current_scope.append(ir.AssertStmt(cond=cond, msg=msg))

    def visit_Return(self, stmt: Return):
        if stmt.value is not None:
            return_value = self.visit(stmt.value)
        else:
            return_value = None
        self.current_scope.append(ir.ReturnStmt(ir.convert(return_value)))

    def visit_Pass(self, stmt: Pass):
        return ir.SeqStmt([])

    def visit_AnnAssign(self, stmt: AnnAssign):
        lhs = stmt.target
        rhs = self.visit(stmt.value) if stmt.value else None
        assert isinstance(lhs, (Name, Attribute, Subscript))
        if isinstance(lhs, (Attribute, Subscript)):
            msg = 'Hidet do not support annotation for expression like "x.y" or "x[y]"'
            raise HidetProgramError(self, stmt.annotation, msg)
        self.process_assign(lhs, rhs, stmt.annotation)

    def visit_While(self, stmt: While):
        if len(stmt.orelse) > 0:
            raise HidetProgramError(self, stmt.orelse[0], 'Hidet does not support else for while statement.')
        cond = self.visit(stmt.test)
        with self.scope() as while_scope:
            for body_stmt in stmt.body:
                self.visit(body_stmt)
        while_stmt = ir.WhileStmt(cond, while_scope.flush_stmts())
        self.current_scope.append(while_stmt)

    def visit_Break(self, stmt: Break):
        self.current_scope.append(ir.BreakStmt())

    def visit_Continue(self, stmt: Continue):
        self.current_scope.append(ir.ContinueStmt())

    def visit_With(self, stmt: With):
        raise HidetProgramError(self, stmt, 'Hidet currently do not support with statement.')

    def visit_Lambda(self, expr: Lambda):
        raise HidetProgramError(self, expr, 'Hidet currently do not support lambda expression.')

    def process_generator(self, elt, generators: list[comprehension]) -> list:
        if len(generators) == 0:
            return [self.visit(elt)]
        else:
            generator = generators[0]
            if generator.is_async:
                raise HidetProgramError(self, generator, 'Hidet currently do not support async generator.')
            assert isinstance(generator, comprehension)
            iterator = self.visit(generator.iter)
            names: list[str] = []
            if isinstance(generator.target, Name):
                names = [generator.target.id]
            elif isinstance(generator.target, Tuple):
                for target in generator.target.elts:
                    if not isinstance(target, Name):
                        raise HidetProgramError(
                            self, target, "Hidet currently only support binding a single name or a tuple of names"
                        )
                    names.append(target.id)
            else:
                raise HidetProgramError(
                    self,
                    generator.target,
                    'Hidet do not support generator target with type {}.'.format(type(generator.target)),
                )
            result = []
            for it in iterator:
                if len(names) == 1:
                    self.current_scope.define_host_var(names[0], it)
                else:
                    if len(names) != len(it):
                        raise HidetProgramError(
                            self, generator, "Can not unpack {} values to {} names.".format(len(it), len(names))
                        )
                    for name, value in zip(names, it):
                        self.current_scope.define_host_var(name, value)
                if not all(self.visit(cond) for cond in generator.ifs):
                    continue
                result.extend(self.process_generator(elt, generators[1:]))
            return result

    def visit_ListComp(self, expr: ListComp):
        return self.process_generator(expr.elt, expr.generators)

    def visit_DictComp(self, expr: DictComp):
        kv_pair = Tuple()
        kv_pair.elts = [expr.key, expr.value]
        kv_pairs = self.process_generator(kv_pair, expr.generators)
        return {k: v for k, v in kv_pairs}

    def visit_SetComp(self, expr: SetComp):
        values = self.process_generator(expr.elt, expr.generators)
        return set(values)

    def visit_GeneratorExp(self, expr: GeneratorExp):
        return self.process_generator(expr.elt, expr.generators)

    def visit_Nonlocal(self, stmt: Nonlocal):
        pass

    def visit_Global(self, stmt: Global):
        pass
