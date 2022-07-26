from __future__ import annotations
from types import FunctionType

import inspect
from typing import Optional, Dict, Any, Union
import warnings
import os.path
import ast
from hidet.ir.builders import FunctionBuilder
from ast import AST, Module
import astunparse

# statements
from ast import FunctionDef, Return, Assign, AnnAssign, AugAssign, For, While, If, With, Assert, Expr, Pass, Break, Continue

# expressions
from ast import Constant, Num, Str
from ast import BoolOp, BinOp, UnaryOp, Lambda, IfExp, Compare, Call, Attribute, Subscript, Starred, Name, Tuple, Slice, ExtSlice, List

# expr context
from ast import Load, Store, Del

# arithmatic and bitwise operators
from ast import UAdd, USub, Add, Sub, Mult, Div, FloorDiv, Mod, Pow, LShift, RShift, BitOr, BitXor, BitAnd, Invert

# bool and compare operators
from ast import Not, And, Or, Eq, NotEq, Lt, LtE, Gt, GtE

from ast import Index

from hidet import ir
from hidet.ir import Var
from hidet.utils import red, cyan, green, bold, blue


class AstNotSupported(Exception):
    pass


class HidetProgramError(Exception):
    def __init__(self, translator: PythonAstFunctor, obj: Union[AST, ast.expr, ast.arg, ast.stmt], msg: str):
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
        lines.append('  File {file}:{line}:{column}:'.format(file=os.path.abspath(self.file), line=self.lineno - 1, column=self.column))
        lines.append('    {msg}'.format(msg=blue(self.msg)))
        if source_line:
            lines.append(source_line)
            lines.append(' ' * self.column + bold(red('^')))
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
        return visitor(node)

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


class Scope:
    def __init__(self, parent: Scope):
        self.parent: Scope = parent
        self.name2var: Dict[str, Var] = {}
        self.name2host_var: Dict[str, Any] = {}
        self.stmts: list[ir.Stmt] = []
        self.attributes: dict[str, Any] = {}

    def define(self, name: str, v: Var):
        self.name2var[name] = v

    def define_host_var(self, name: str, v: Any):
        self.name2host_var[name] = v

    def lookup(self, name: str, search_parents=True) -> Optional[Union[Var, Any]]:
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


class ScopeContext:
    def __init__(self, scopes: list[Scope]):
        self.scopes = scopes

    def __enter__(self) -> Scope:
        parent_scope = self.scopes[-1] if len(self.scopes) > 0 else None
        new_scope = Scope(parent_scope)
        self.scopes.append(new_scope)
        return new_scope

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()


class PythonToHidetTranslator(PythonAstFunctor):
    def __init__(self, file, start_lineno, start_column, env, func_annotations):
        super().__init__(file, start_lineno, start_column)
        self.env: Dict[str, Any] = env
        self.func_annotations: Dict[str, Any] = func_annotations

        self.fb: Optional[FunctionBuilder] = None
        self.scopes: list[Scope] = []

    def scope(self) -> ScopeContext:
        return ScopeContext(self.scopes)

    @property
    def current_scope(self) -> Scope:
        return self.scopes[-1]

    def process_assign(self, lhs: Union[Attribute, Subscript, Name], rhs: Union[ast.expr, ir.Expr, list[ast.expr]]):
        if isinstance(rhs, ast.expr):
            value = self.visit(rhs)
        elif isinstance(rhs, ir.Expr):
            value = rhs
        elif isinstance(rhs, list):
            value = [self.visit(v) for v in rhs]
        else:
            raise ValueError(rhs)

        if isinstance(lhs, Name):
            var = self.current_scope.lookup(lhs.id)
            if var is None:
                var_name = lhs.id
                if isinstance(value, (ir.TaskMapping, ir.DataLayout)):
                    self.current_scope.define_host_var(var_name, value)
                else:
                    if isinstance(value, ir.TypeNode):
                        # a = tensor('shared', 'float32', [3, 4])
                        var_type = value
                        init_value = None
                    else:
                        # a = 5
                        var_type = ir.infer_type(value)
                        init_value = value
                    var = Var(hint=var_name, type=var_type)
                    self.current_scope.append(ir.DeclareStmt(var, init=init_value))
                    self.current_scope.define(name=var_name, v=var)
            else:
                self.current_scope.append(ir.AssignStmt(var, value=value))
        elif isinstance(lhs, Subscript):
            base = self.visit(lhs.value)
            indices = self.visit(lhs.slice)
            if not isinstance(indices, list):
                indices = [indices]
            self.current_scope.append(ir.BufferStoreStmt(buf=base, indices=indices, value=value))
        elif isinstance(lhs, Attribute):
            from hidet.lang import attr
            # attr.cuda_block_dim = ...
            lhs_base = self.visit(lhs.value)
            if lhs_base is attr:
                attr_name = lhs.attr
                self.current_scope.annotate(attr_name, value)
            else:
                raise HidetProgramError(self, lhs, 'Invalid assignment.')
        else:
            raise HidetProgramError(self, lhs, 'Cannot recognize "{}" as left side of assignment.'.format(type(lhs).__name__))

    def visit_Module(self, module: Module):
        if len(module.body) != 1 or not isinstance(module.body[0], FunctionDef):
            raise ValueError('The module expects to have only one function definition statement, got\n{}'.format(ast.unparse(module)))
        return self.visit(module.body[0])

    def visit_FunctionDef(self, func_def: FunctionDef):
        func_name = func_def.name
        func_params = []
        with self.scope() as scope:
            # process function arguments
            args: ast.arguments = func_def.args
            if args.vararg is not None:
                raise HidetProgramError(self, args.vararg, 'Hidet program does not support "*args" arguments.')
            if len(args.kwonlyargs) != 0:
                raise HidetProgramError(self, args.kwonlyargs[0], 'Hidet program does not support "*kwargs" arguments.')
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
                if isinstance(arg_type, ir.TensorType):
                    arg_type = ir.TensorPointerType(scope='global', dtype=arg_type.scalar_type, shape=arg_type.shape, layout=arg_type.layout)
                param_var = Var(hint=arg_name, type=arg_type)
                func_params.append(param_var)
                scope.define(arg_name, param_var)

            # process function body
            for stmt in func_def.body:
                self.visit(stmt)

        if 'cuda_grid_dim' not in scope.attributes:
            raise HidetProgramError(self, func_def, "cuda requires to specify 'attr.cuda_grid_dim' attribute to define the number of thread blocks to launch.")
        if 'cuda_block_dim' not in scope.attributes:
            raise HidetProgramError(self, func_def, "cuda requries to specify 'attr.cuda_block_dim' attribute to define the number of threads per block.")

        return ir.Function(
            name=func_name,
            params=func_params,
            body=scope.flush_stmts(),
            ret_type=ir.VoidType(),
            kind='cuda_kernel',
            local_vars=[],          # todo: fill the following parameters
            local_const_vars=[],
            extern_vars=ir.primitives.cuda.vars.get_all_primitive_vars(),
            attrs={
                'cuda_grid_dim': scope.attributes['cuda_grid_dim'],
                'cuda_block_dim': scope.attributes['cuda_block_dim']
            }
        )

    def visit_Assign(self, stmt: Assign):
        if len(stmt.targets) > 1:
            raise HidetProgramError(self, stmt, 'Hidet does not support syntax like "a = b = 1".')
        target = stmt.targets[0]
        value = stmt.value
        if isinstance(target, (Tuple, List)):
            lhs_list = target.elts
        else:
            lhs_list = [target]

        if isinstance(value, (Tuple, List)):
            rhs_list = value.elts
        else:
            rhs_list = [value]

        if len(lhs_list) == len(rhs_list):
            for lhs, rhs in zip(lhs_list, rhs_list):
                self.process_assign(lhs, rhs)
        elif len(lhs_list) == 1:
            lhs = lhs_list[0]
            assert isinstance(lhs, (Subscript, Name, Attribute))
            self.process_assign(lhs, rhs_list)
        elif len(rhs_list) == 1:
            raise HidetProgramError(self, stmt, 'Hidet does not support unpacking.')
        else:
            raise HidetProgramError(self, stmt, 'Can not assign {} elements to {} elements.'.format(len(rhs_list), len(lhs_list)))

    def visit_Name(self, expr: Name):
        if isinstance(expr.ctx, Store):
            raise ValueError('Internal, please deal with all Store behavior in parent nodes like Assign.')
        elif isinstance(expr.ctx, Load):
            name: str = expr.id
            var: Optional[Var] = self.current_scope.lookup(name)
            if var is None:
                if name in self.env:
                    # access external variable, such as thread_x
                    return self.env[name]
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
            assert isinstance(expr.op, Mult)
            return lhs * rhs
        elif isinstance(lhs, ir.TaskMapping) and isinstance(rhs, ir.TaskMapping):
            assert isinstance(expr.op, Mult)
            return lhs * rhs
        else:
            op_dict = {
                Add: ir.Add,
                Sub: ir.Sub,
                Mult: ir.Multiply,
                Div: ir.Div,
                FloorDiv: ir.Div,   # we treat Div and FloorDiv equivalently with the same semantics as in C/C++
                Mod: ir.Mod
            }
            if isinstance(expr.op, Pow):
                return ir.primitives.pow(lhs, rhs)
            elif type(expr.op) in op_dict:
                return op_dict[type(expr.op)](lhs, rhs)
            else:
                raise HidetProgramError(self, expr, 'Currently, we do not support {} operator.'.format(type(expr.op).__name__))

    def visit_BoolOp(self, expr: BoolOp):
        values = [self.visit(v) for v in expr.values]
        if isinstance(expr.op, And):
            return ir.And.join_list(values)
        else:
            assert isinstance(expr.op, Or)
            return ir.Or.join_list(values)

    def visit_Compare(self, expr: Compare):
        front = self.visit(expr.left)
        op_dict = {
            And: ir.And,
            Or: ir.Or,
            Eq: ir.Equal,
            Gt: lambda a, b: ir.LessThan(b, a),
            Lt: ir.LessThan,
            GtE: lambda a, b: ir.LessEqual(b, a),
            LtE: ir.LessEqual
        }
        cond = None
        comparators = [self.visit(v) for v in expr.comparators]
        for op, current in zip(expr.ops, comparators):
            cur_cond = op_dict[type(op)](front, current)
            cond = ir.And(cond, cur_cond) if cond is not None else cur_cond
            front = current
        return cond

    def visit_UnaryOp(self, expr: UnaryOp):
        pass

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
        if isinstance(stmt.target, (List, Tuple)):
            iter_targets = stmt.target.elts
        else:
            iter_targets = [stmt.target]
        assert all(isinstance(v, Name) for v in iter_targets)
        loop_vars: list[Var] = []
        for target in iter_targets:
            assert isinstance(target, Name)
            loop_vars.append(Var(target.id, type=ir.scalar_type('int32')))

        # construct for body
        if isinstance(stmt.iter, Call) and isinstance(stmt.iter.func, Name) and stmt.iter.func.id == 'range':
            # case 1:
            #   for i in range(...):
            #     ...
            # Will be translated to ForStmt
            call = stmt.iter

            # get extent
            if len(call.args) > 1:
                raise NotImplementedError('Current we only support range(extent), will add range(start, stop, step) later.')
            extent = self.visit(call.args[0])

            with self.scope() as for_scope:
                assert len(loop_vars) == 1
                for_scope.define(name=loop_vars[0].hint, v=loop_vars[0])
                for s in stmt.body:
                    self.visit(s)
            self.current_scope.append(ir.ForStmt(
                loop_var=loop_vars[0],
                extent=extent,
                body=for_scope.flush_stmts()
            ))
        else:
            # case 2:
            #  for a, b in row_spatial(3, 4).on(thread_x):
            #    ...
            # Will be translated to ForTaskStmt
            raise NotImplementedError()
        if len(stmt.orelse) > 0:
            raise HidetProgramError(self, stmt.orelse[0], 'Hidet does not support else clause in for loop.')

    def visit_AugAssign(self, stmt: AugAssign):
        var_value = self.visit(stmt.target)
        value = self.visit(stmt.value)
        op_dict = {
            Add: ir.Add,
            Sub: ir.Sub,
            Mult: ir.Multiply,
            Div: ir.Div,
            FloorDiv: ir.Div,   # we treat Div and FloorDiv equivalently with the same semantics as in C/C++
            Mod: ir.Mod
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
            raise HidetProgramError(self, expr, 'Can not access attribute.')

    def visit_IfExp(self, expr: IfExp):
        cond = self.visit(expr.test)
        then_expr = self.visit(expr.body)
        else_expr = self.visit(expr.orelse)
        return ir.expr.if_then_else(cond, then_expr, else_expr)

    def visit_Expr(self, stmt: Expr):
        value = self.visit(stmt.value)
        if isinstance(value, ir.expr.Call):
            self.current_scope.append(ir.EvaluateStmt(value))
        else:
            raise HidetProgramError(self, stmt, 'Can not recognize expression statement.')

    def visit_Call(self, expr: Call):
        func = self.visit(expr.func)
        args = [self.visit(arg) for arg in expr.args]
        kwargs = {kwarg.arg: self.visit(kwarg.value) for kwarg in expr.keywords}
        if isinstance(func, FunctionType):
            return func(*args, **kwargs)
        elif isinstance(func, ir.Expr):
            if len(kwargs) > 0:
                raise HidetProgramError(self, expr, 'Hidet do not support call with keyword.')
            return ir.expr.Call(func, args)
        else:
            raise ValueError('Can not recognize callee {}'.format(func))

    def visit_Starred(self, expr: Starred):
        raise HidetProgramError(self, expr, 'Hidet do not support unpack operator.')

    def visit_ExtSlice(self, expr: ExtSlice):
        return [self.visit(v) for v in expr.dims]

    def visit_Slice(self, expr: Slice):
        return slice(
            self.visit(expr.lower) if expr.lower is not None else None,
            self.visit(expr.upper) if expr.upper is not None else None,
            self.visit(expr.step) if expr.step is not None else None
        )

