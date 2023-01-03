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
import builtins

from typing import Optional, Dict, Any, Union
import os.path
import ast
from ast import Module

# statements
from ast import FunctionDef, Return, Assign, AnnAssign, AugAssign, For, While, If, With, Assert, Expr, Pass, Break
from ast import Continue

# expressions
from ast import Constant, Num, Str, NameConstant
from ast import BoolOp, BinOp, UnaryOp, Lambda, IfExp, Compare, Call, Attribute, Subscript, Starred, Name, Tuple, Slice
from ast import ExtSlice, List

# expr context
from ast import Load, Store, Del

# arithmetic and bitwise operators
from ast import UAdd, USub, Add, Sub, Mult, Div, FloorDiv, Mod, Pow, BitOr, BitXor, BitAnd, Invert

# bool and compare operators
from ast import Not, And, Or, Eq, NotEq, Lt, LtE, Gt, GtE

from ast import Index

import astunparse
from hidet import ir
from hidet.ir.expr import Var
from hidet.ir.stmt import DeclareScope
from hidet.ir.functors import simplify
from hidet.ir.builders import FunctionBuilder
from hidet.utils import red, bold, blue, str_indent
import hidet.lang.attr
from hidet.lang.type_utils import TypeDecorator


class HidetProgramError(Exception):
    def __init__(
        self,
        translator: PythonAstFunctor,
        obj: Union[ast.AST, ast.expr, ast.arg, ast.stmt, ast.Attribute, ast.Name],
        msg: str,
    ):
        super().__init__()
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
    def __init__(self, file: str, start_lineno: int, start_column: int):
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column

    def __call__(self, node):
        return self.visit(node)

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        if hasattr(self, method):
            visitor = getattr(self, method)
        else:
            msg = 'The AST node {} does not support in HidetScript.'.format(node.__class__.__name__)
            raise HidetProgramError(self, node, msg)

        try:
            return visitor(node)
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


HostTypes = (ir.TaskMapping, ir.DataLayout, float, int, type(None))


class Scope:
    def __init__(self, parent: Optional[Scope]):
        self.parent: Optional[Scope] = parent
        self.name2var: Dict[str, Var] = {}
        self.name2host_var: Dict[str, Any] = {}
        self.stmts: list[ir.Stmt] = []
        self.attributes: dict[str, Any] = {}

    def define_var(self, name: str, v: Var):
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
        self.scopes: list[Scope] = []

    def __enter__(self) -> Scope:
        if len(self.scopes) == 0:
            parent = None
        else:
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
        host_var_types = (ir.TaskMapping, ir.DataLayout, ir.TensorSlice, ir.Function, str, list, tuple)
        allowed_types = (ir.Expr, ir.TypeNode, TypeDecorator, float, int, str, type(None))
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
                        if isinstance(rhs, ir.TypeNode):
                            var_type = rhs
                        elif isinstance(rhs, TypeDecorator):
                            var_type = rhs.decorated_type
                            is_static = rhs.is_static
                            scope = rhs.scope
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
            # example: attr.cuda_block_dim = 16, 16
            lhs_base = self.visit(lhs.value)
            if lhs_base is hidet.lang.attr:
                attr_name = lhs.attr
                if attr_name in ['cuda_block_dim', 'cuda_grid_dim', 'cuda_dynamic_smem_bytes']:
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

    def visit_FunctionDef(self, func_def: FunctionDef):
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements, import-outside-toplevel
        from hidet.ir.primitives.cuda.vars import thread_idx, block_idx

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
                    arg_name = arg.arg
                    if arg_name not in self.func_annotations:
                        raise HidetProgramError(self, arg, 'Hidet expects type annotation for each function argument.')
                    arg_type = self.func_annotations[arg_name]
                    if isinstance(arg_type, ir.TypeNode):
                        if isinstance(arg_type, ir.TensorType):
                            # we automatically change the tensor type of argument to a tensor pointer type.
                            arg_type = ir.TensorPointerType(
                                dtype=arg_type.dtype, shape=arg_type.shape, layout=arg_type.layout
                            )
                    elif isinstance(arg_type, TypeDecorator):
                        arg_type = arg_type.decorated_type
                        if isinstance(arg_type, ir.TensorType):
                            # we automatically change the tensor type of argument to a tensor pointer type.
                            arg_type = ir.TensorPointerType(
                                dtype=arg_type.dtype, shape=arg_type.shape, layout=arg_type.layout
                            )
                    elif arg_type in [int, float]:
                        type_dict = {int: ir.data_type('int32'), float: ir.data_type('float32')}
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
                ret_type = self.visit(func_def.returns)
                if not isinstance(ret_type, ir.TypeNode):
                    raise HidetProgramError(self, func_def.returns, 'Expect a type of function return value.')

            # get function attributes
            func_attrs: Dict[str, Any] = scope.attributes.copy()
            if 'func_kind' in func_attrs:
                func_kind = func_attrs['func_kind']
            elif 'cuda_grid_dim' in func_attrs or 'cuda_block_dim' in func_attrs:
                if not all(name in func_attrs for name in ['cuda_grid_dim', 'cuda_block_dim']):
                    raise HidetProgramError(
                        self, func_def, 'CUDA kernel expects to have both attrs.cuda_grid_dim and attrs.cuda_block_dim.'
                    )
                func_kind = 'cuda_kernel'
            else:
                func_kind = 'cuda_device'
            func_name = func_attrs.get('func_name', func_def.name)

        return ir.Function(
            name=func_name,
            params=func_params,
            body=scope.flush_stmts(),
            ret_type=ret_type,
            kind=func_kind,
            extern_vars=[
                thread_idx('x'),
                thread_idx('y'),
                thread_idx('z'),
                block_idx('x'),
                block_idx('y'),
                block_idx('z'),
            ],
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
        elif isinstance(lhs, list) and isinstance(rhs, list):
            assert isinstance(expr.op, Add)
            return lhs + rhs
        elif isinstance(lhs, tuple) and isinstance(rhs, tuple):
            assert isinstance(expr.op, Add)
            return lhs + rhs
        elif isinstance(lhs, (ir.Expr, float, int)) and isinstance(rhs, (ir.Expr, float, int)):
            op_dict = {
                Add: ir.Add,
                Sub: ir.Sub,
                Mult: ir.Multiply,
                Div: ir.Div,
                FloorDiv: ir.Div,  # we treat Div and FloorDiv equivalently with the same semantics as in C/C++
                Mod: ir.Mod,
                BitXor: ir.BitwiseXor,
                BitOr: ir.BitwiseOr,
                BitAnd: ir.BitwiseAnd,
            }
            # pylint: disable=import-outside-toplevel
            from hidet.ir import primitives

            if isinstance(expr.op, Pow):
                return primitives.pow(lhs, rhs)
            elif type(expr.op) in op_dict:
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
            return ir.LogicalAnd.join_list(values)
        else:
            assert isinstance(expr.op, Or)
            return ir.LogicalOr.join_list(values)

    def visit_Compare(self, expr: Compare):
        front = self.visit(expr.left)
        op_dict = {
            And: ir.LogicalAnd,
            Or: ir.LogicalOr,
            Eq: ir.Equal,
            Gt: lambda a, b: ir.LessThan(b, a),  # pylint: disable=arguments-out-of-order
            Lt: ir.LessThan,
            GtE: lambda a, b: ir.LessEqual(b, a),  # pylint: disable=arguments-out-of-order
            LtE: ir.LessEqual,
            NotEq: ir.NotEqual,
        }
        cond = None
        comparators = [self.visit(v) for v in expr.comparators]
        for op, current in zip(expr.ops, comparators):
            cur_cond = op_dict[type(op)](front, current)
            cond = ir.LogicalAnd(cond, cur_cond) if cond is not None else cur_cond
            front = current
        return cond

    def visit_UnaryOp(self, expr: UnaryOp):
        value = self.visit(expr.operand)
        if isinstance(expr.op, Not):
            # not v
            return ir.LogicalNot(value)
        elif isinstance(expr.op, Invert):
            # there are two cases for a ~ operator: ~something
            # case 1: get the address of an expression
            # case 2: get the pointer type that points to the given type
            from hidet.ir.expr import Address
            from hidet.ir.type import TypeNode

            if isinstance(value, TypeNode):
                return ~value
            else:
                return Address(value)
        elif isinstance(expr.op, UAdd):
            # +v
            return value
        elif isinstance(expr.op, USub):
            # -v
            return ir.Neg(value)
        else:
            raise HidetProgramError(self, expr, 'Can not recognize unary operator.')

    def visit_If(self, stmt: If):
        cond = self.visit(stmt.test)
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

        loop_vars: list[Var] = []
        host_vars: Dict[str, Any] = {}

        def visit_body() -> ir.Stmt:
            with self.scope() as for_scope:
                for var in loop_vars:
                    for_scope.define_var(name=var.hint, v=var)
                for name, value in host_vars.items():
                    for_scope.define_host_var(name, value)
                for s in stmt.body:
                    self.visit(s)
            return for_scope.flush_stmts()

        def declare_loop_vars(num: int):
            p = len(iter_targets)
            q = num

            # for i_1, ..., i_p in grid(n_1, ..., n_q):
            #    ...
            # we mush have either
            #  1. p == q,
            #  2. or p == 1 and q > 1

            if p not in (1, q):
                raise HidetProgramError(self, stmt, f'Expect {num} loop variables, but got {len(iter_targets)}.')

            if p == q:
                for target in iter_targets:
                    loop_vars.append(Var(target.id, type=ir.data_type('int32')))
            else:
                name = iter_targets[0].id
                for i in range(q):
                    loop_vars.append(Var(f'_{name}_{i}', type=ir.data_type('int32')))
                host_vars[name] = list(loop_vars)

        # construct for body
        if (
            isinstance(stmt.iter, Call)
            and isinstance(stmt.iter.func, Name)
            and self.visit(stmt.iter.func) is builtins.range
        ):  # and stmt.iter.func.id == 'range':
            # case 1:
            #   for i in range(...):
            #     ...
            # Will be translated to a single for loop (i.e., ForStmt).
            call = stmt.iter

            # get extent
            if len(call.args) > 1:
                msg = 'Current we only support range(extent), will add range(start, stop, step) when needed.'
                raise NotImplementedError(msg)
            declare_loop_vars(num=1)
            extent = self.visit(call.args[0])
            self.current_scope.append(ir.ForStmt(loop_var=loop_vars[0], extent=extent, body=visit_body()))
        elif (
            isinstance(stmt.iter, Call)
            and isinstance(stmt.iter.func, Name)
            and self.visit(stmt.iter.func) is hidet.lang.grid
        ):
            # case 2:
            #  for i, j in grid(3, 4):
            #    ...
            # Will be translated to nested for loops (i.e., ForStmt).
            call = stmt.iter
            extents = [self.visit(arg) for arg in call.args]
            declare_loop_vars(num=len(extents))
            body = visit_body()
            for loop_var, extent in zip(reversed(loop_vars), reversed(extents)):
                body = ir.ForStmt(loop_var=loop_var, extent=extent, body=body)
            self.current_scope.append(body)
        elif isinstance(stmt.iter, Call) and isinstance(stmt.iter.func, Attribute) and stmt.iter.func.attr == 'on':
            # case 3:
            #  for a, b in row_spatial(3, 4).on(thread_x):
            #    ...
            # Will be translated to ForTaskStmt
            if len(stmt.iter.args) != 1:
                raise HidetProgramError(self, stmt.iter, 'Expect a single expression representing worker index.')
            worker = ir.convert(self.visit(stmt.iter.args[0]))
            mapping = self.visit(stmt.iter.func.value)
            if not isinstance(mapping, ir.TaskMapping):
                raise HidetProgramError(self, stmt.iter.func.value, 'Expect task mapping here.')
            declare_loop_vars(num=len(mapping.task_shape))
            self.current_scope.append(
                ir.ForTaskStmt(loop_vars=loop_vars, mapping=mapping, worker=worker, body=visit_body())
            )
        else:
            msg = (
                'Cannot recognize the for loop statement. Currently, we support two types of for loop statements:\n'
                '  for i in range(extent):\n'
                '    body(i)\n'
                'and\n'
                '  for i, j in mapping.on(worker):\n'
                '    body(i, j)\n'
                '  (here the mapping can be arbitrary task mapping, or their composition with arbitrary dimensions).'
            )
            raise HidetProgramError(self, stmt.iter, msg)

        if len(stmt.orelse) > 0:
            raise HidetProgramError(self, stmt.orelse[0], 'Hidet does not support else clause in for loop.')

    def visit_AugAssign(self, stmt: AugAssign):
        if isinstance(stmt.target, Name):
            target = Name(stmt.target.id, Load())
            var_value = self.visit(target)
        else:
            var_value = self.visit(stmt.target)
            # raise NotImplementedError()
        value = self.visit(stmt.value)
        op_dict = {
            Add: ir.Add,
            Sub: ir.Sub,
            Mult: ir.Multiply,
            Div: ir.Div,
            FloorDiv: ir.Div,  # we treat Div and FloorDiv equivalently with the same semantics as in C/C++
            Mod: ir.Mod,
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
        kwargs = {kwarg.arg: self.visit(kwarg.value) for kwarg in expr.keywords}

        if isinstance(func, types.FunctionType):
            # call python function
            return func(*args, **kwargs)
        elif isinstance(func, types.MethodType):
            # call python class method
            return func(*args, **kwargs)
        elif isinstance(func, ir.Expr):
            from hidet.ir.functors import infer_type

            # call hidet function
            if len(kwargs) > 0:
                raise HidetProgramError(self, expr, 'Hidet do not support call with keyword.')
            func_type: ir.FuncType = infer_type(func)
            assert isinstance(func_type, ir.FuncType)
            if len(func_type.param_types) != len(args):
                msg = 'The number of parameters of callee and given arguments does not match.'
                raise HidetProgramError(self, expr, msg)
            return ir.Call(func, args)
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
            return ir.Call(func_var, args)
        elif isinstance(func, (types.BuiltinMethodType, types.BuiltinFunctionType)):
            # call python builtin method, such "a string".format(...) or max, min
            from hidet.ir import primitives

            if all(not isinstance(arg, ir.Node) for arg in args):
                # pure python function call
                return func(*args, **kwargs)
            else:
                # overload hidet primitive, such as max, min
                if len(kwargs) > 0:
                    msg = 'Hidet do not support calling builtin function with keyword argument.'
                    raise HidetProgramError(self, expr, msg)
                if func is builtins.max:
                    if len(args) != 2:
                        msg = 'Hidet builtin function "max" takes two arguments.'
                        raise HidetProgramError(self, expr, msg)
                    return primitives.max(args[0], args[1])
                elif func is builtins.min:
                    if len(args) != 2:
                        msg = 'Hidet builtin function "min" takes two arguments.'
                        raise HidetProgramError(self, expr, msg)
                    return primitives.min(args[0], args[1])
                else:
                    raise HidetProgramError(self, expr, 'Currently, do not support calling python builtin function.')
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
        self.current_scope.append(ir.ReturnStmt(return_value))

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
