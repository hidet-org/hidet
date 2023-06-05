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
# pylint: disable=exec-used
# pylint: disable=eval-used
# pylint: disable=dangerous-default-value
# pylint: disable=redefined-outer-name
# pylint: disable=broad-except
# pylint: disable=raise-missing-from

import ast
from ast import AST
import inspect

import typing as tp
from typing import Any, Union, Tuple, List, Dict, Set
import types
import astunparse
from astunparse import Unparser

import torch
import hidet

# from hidet.utils.model_translator.convert import AstTypedNode

from hidet.utils.model_translator.utils import quote_cyan, quote_fail, quote_green, quote_warning

# must do this to load the registery
import hidet.graph.frontend.torch.register_functions
import hidet.graph.frontend.torch.register_methods
import hidet.graph.frontend.torch.register_modules

TYPEDNODE_ATTR = "_typed_node"
STMT_ERROR_ATTR = "_stmt_error_attr"
STMT_WARNING_ATTR = "_stmt_warning_attr"


class AstInternalError(Exception):
    """AstInternalError means a bug"""


class AstRunTimeError(Exception):
    """AstRunTimeError means for user intervention"""


class AstTypedNode:
    def __init__(self) -> None:
        self.type = set()
        self.value = None

    def add(self, val):
        # pylint: disable=broad-except
        if callable(val):
            if is_bound_method(val):
                try:
                    self.type.add(get_unbound_method(val))
                except Exception:
                    self.type.add(val)
            elif inspect.isfunction(val) or inspect.ismethod(val) or inspect.isclass(val):
                self.type.add(val)
            else:
                self.type.add(type(val))
        else:
            self.type.add(type(val))
        self.value = val
        return self


def is_bound_method(meth):
    return callable(meth) and hasattr(meth, '__self__')


def get_unbound_method(meth):
    assert is_bound_method(meth)
    return getattr(type(meth.__self__), meth.__name__)


def hashable(obj) -> bool:
    return hasattr(obj, "__hash__") and obj.__hash__ is not None


def default_should_trace(obj):
    if belong_to_torch(obj) or (not hashable(obj)):
        return False
    if inspect.getmodule(obj) is None:
        return False
    if not hasattr(obj, '__name__'):
        return False
    if obj.__module__ == 'builtins':
        return False
    return True


def get_types(node: AST) -> tp.Union[tp.Set[Any], None]:
    if hasattr(node, TYPEDNODE_ATTR):
        _types = getattr(node, TYPEDNODE_ATTR)
        assert isinstance(_types, AstTypedNode)
        return _types.type
    return None


class StripValues(ast.NodeVisitor):
    """Delete values for deepcopying"""

    def visit(self, node: AST):
        if hasattr(node, TYPEDNODE_ATTR):
            delattr(node, TYPEDNODE_ATTR)
        return super().visit(node)


def get_class_of_method(meth, init_cls):
    assert inspect.isclass(init_cls)
    if hasattr(init_cls, meth.__name__):
        if meth.__name__ in init_cls.__dict__:
            return init_cls
        for inherit in init_cls.__bases__:
            par = get_class_of_method(meth, inherit)
            if par is not None:
                return par
    return None


class RewriteSuper(ast.NodeTransformer):
    """
    Rewrites any occurances of super().foo to super(CurClass, self).foo
    """

    def __init__(self) -> None:
        super().__init__()
        self.class_names = []

    def visit_Call(self, node: ast.Call) -> ast.Call:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == 'super'
            and isinstance(node.func.ctx, ast.Load)
            and len(node.args) == 0
            and len(node.keywords) == 0
        ):
            node.args = [ast.Name(id=self.class_names[-1], ctx=ast.Load()), ast.Name(id='self', ctx=ast.Load())]
        else:
            node = self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.class_names.append(node.name)
        node = self.generic_visit(node)
        self.class_names.pop(-1)
        return node


class InterpreterState:
    def __init__(self) -> None:
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.globals: Dict[str, Any] = {}
        self.local_stack: List[Dict[str, Any]] = []
        # if we are in the context of a class (necessary for super())
        self.marked_global: List[Set[str]] = []

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_warnings(self, node: AST) -> AST:
        if len(self.warnings) > 0:
            self.warnings, warnings = [], self.warnings
            setattr(node, STMT_WARNING_ATTR, warnings)
        return node

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_errors(self, node: AST) -> AST:
        if len(self.errors) > 0:
            self.errors, errors = [], self.errors
            setattr(node, STMT_ERROR_ATTR, errors)
        return node

    @property
    def locals(self):
        return self.local_stack[-1]

    # pylint: disable=dangerous-default-value
    def push_stack(self, node: ast.arguments, arg_values: List[Any] = [], kwarg_values: Dict[str, Any] = {}):
        """binds arguments to a new local stack, this happens when calling a function"""
        args = {}
        self.local_stack.append(args)
        self.marked_global.append(set())

        if node.vararg is not None:
            args[node.vararg.arg] = arg_values[len(node.args) :]
            arg_values = arg_values[: len(node.args)]
        defaults = node.defaults.copy()
        for a in node.args:
            if len(arg_values) > 0:
                args[a.arg] = arg_values.pop(0)
            elif a.arg in kwarg_values:
                val = kwarg_values.pop(a.arg)
                args[a.arg] = val
            elif len(defaults) > 0:
                default = defaults.pop(0)
                args[a.arg] = self.exec_node(default)
            else:
                raise Exception("Not enough arguments")

        kw_defaults = node.kw_defaults.copy()
        for kw in node.kwonlyargs:
            if kw.arg in kwarg_values:
                args[kw.arg] = kwarg_values[kw.arg]
            elif len(kw_defaults) > 0:
                default = kw_defaults.pop(0)
                args[kw.arg] = self.exec_node(default)
            else:
                raise Exception("Not enough keyword arguments")

    def pop_stack(self) -> Any:
        """when function call ends, pops the latest stack, returning what the function returned as result"""
        last_stack = self.local_stack.pop(-1)
        self.marked_global.pop()
        if '__return__' in last_stack:
            return last_stack['__return__']
        else:
            return None

    def set_return(self, val):
        assert '__return__' not in self.locals
        self.locals['__return__'] = val

    def mark_globals(self, node: ast.Global):
        for i in node.names:
            self.marked_global[-1].add(i)

    def get_unique_local_var(self, base: str) -> str:
        variable = base
        while variable in self.locals or variable in self.marked_global:
            variable += "_"
        return variable

    def exec_node(self, node: AST) -> Any:
        exec_str = astunparse.unparse(node)
        if isinstance(node, ast.expr):
            try:
                return eval(exec_str, self.globals, self.locals)
            except Exception as e:
                raise AstRunTimeError(f"unable to evaluate node {node}, {node.lineno} due to {e}")
        else:
            try:
                exec(exec_str, self.globals, self.locals)
                return None
            except Exception as e:
                raise AstRunTimeError(f"unable to execute node {node}, {node.lineno} due to {e}")

    def get_identifer(self, name: str) -> Any:
        if name in self.locals:
            return self.locals[name]
        elif name in self.globals:
            return self.globals[name]
        elif hasattr(__builtins__, name):
            return getattr(__builtins__, name)
        else:
            raise AstRunTimeError(f"unable to get identifier {name}")

    def set_identifier(self, name: str, val: Any):
        if not name in self.marked_global[-1]:
            self.locals[name] = val
        else:
            self.globals[name] = val


class AstTrace:
    def __init__(self) -> None:
        self.traced: Dict[Union[tp.Callable, type], AST] = {}
        self.method_map: Dict[tp.Callable, type] = {}

    def can_trace(self, obj):
        return hashable(obj) and hasattr(obj, '__name__')

    def add_function(self, obj):
        assert inspect.isfunction(obj)
        assert hashable(obj)
        assert obj not in self.traced
        ast_node = ast.parse(inspect.getsource(obj)).body[0]
        assert isinstance(ast_node, ast.FunctionDef)
        self.traced[obj] = ast_node

    def add_class(self, obj):
        """This does not recursively add all the inherited bases"""
        assert inspect.isclass(obj)
        assert hashable(obj)
        assert obj not in self.traced
        ast_node: ast.ClassDef = ast.parse(inspect.getsource(obj)).body[0]
        assert isinstance(ast_node, ast.ClassDef)
        for fn in ast_node.body:
            if isinstance(fn, ast.FunctionDef):
                fn_obj = getattr(obj, fn.name)
                if isinstance(fn_obj, property):
                    self.method_map[fn_obj.fget] = obj
                elif inspect.isfunction(fn_obj):
                    self.method_map[fn_obj] = obj
        self.traced[obj] = ast_node

    def add_trace(self, obj):
        if self.can_trace(obj):
            if not (obj in self.traced or obj in self.method_map):
                if inspect.isfunction(obj):
                    self.add_function(obj)
                elif inspect.isclass(obj):
                    self.add_class(obj)
                elif is_bound_method(obj):
                    parent_class = type(obj.__self__)
                    self.add_trace(parent_class)

    def get_trace(self, obj: Any) -> AST:
        if self.can_trace(obj):
            # is function or class
            if obj in self.traced:
                return self.traced[obj]

            # is class method
            if is_bound_method(obj):
                parent = type(obj.__self__)
                parent = get_class_of_method(obj, parent)
            elif obj in self.method_map:
                parent = self.method_map[obj]
            elif isinstance(obj, property):
                parent = self.method_map[obj.fget]
            else:
                return None
            if parent not in self.traced:
                return None
            class_ast: ast.ClassDef = self.traced[parent]
            assert isinstance(class_ast, ast.ClassDef)
            for fn in class_ast.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == obj.__name__:
                    return fn
        return None

    def set_trace(self, obj: Any, ast_node: AST) -> bool:
        if self.can_trace(obj):
            # is function or class
            if obj in self.traced:
                if isinstance(ast_node, ast.FunctionDef):
                    assert inspect.isfunction(obj)
                    assert isinstance(self.traced[obj], ast.FunctionDef), "setting different ast"
                    self.traced[obj] = ast_node
                    return True
                elif isinstance(ast_node, ast.ClassDef):
                    assert inspect.isclass(obj)
                    assert isinstance(self.traced[obj], ast.ClassDef), "setting different ast"
                    self.traced[obj] = ast_node
                    return True
                else:
                    return False

            # is class method
            if is_bound_method(obj):
                parent = type(obj.__self__)
            elif obj in self.method_map:
                parent = self.method_map[obj]
            elif isinstance(obj, property) and obj.fget in self.method_map:
                parent = self.method_map[obj.fget]
            else:
                return False
            if parent not in self.traced:
                return False
            class_ast: ast.ClassDef = self.traced[parent]
            assert isinstance(class_ast, ast.ClassDef)
            assert isinstance(ast_node, ast.FunctionDef)
            for i, fn in enumerate(class_ast.body):
                if isinstance(fn, ast.FunctionDef) and fn.name == obj.__name__:
                    class_ast.body[i] = ast_node
                    return True
        return False


class AstInterpreter:
    def __init__(self) -> None:
        self.state = InterpreterState()
        self.trace = AstTrace()
        self.trace_blacklist = set()

        self.returned = False
        self.inner_break = False
        self.indent_level = 0

    def trace_msg(self, msg):
        # print("    " * self.indent_level + msg)
        pass

    def indent(self):
        self.indent_level += 1

    def unindent(self):
        self.indent_level = max(0, self.indent_level - 1)

    def add_val(self, node: AST, val) -> AST:
        if val is None:
            return node
        if not hasattr(node, TYPEDNODE_ATTR):
            setattr(node, TYPEDNODE_ATTR, AstTypedNode())
        type_node = getattr(node, TYPEDNODE_ATTR)
        assert isinstance(type_node, AstTypedNode)
        type_node.add(val)
        return node

    def visit(self, node: AST) -> Tuple[AST, Any]:
        method = 'visit_' + node.__class__.__name__
        if not hasattr(self, method):
            # self.add_warning(f"node {type(node)} not implemented yet, resolve manually.")
            # print(f"{method} not found, executing default")
            val = self.state.exec_node(node)
            return self.add_val(node, val), val
        visitor = getattr(self, method)
        return visitor(node)

    def generic_visit(self, node: AST) -> AST:
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        if not isinstance(value, AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    #######################################################################
    ### Expressions #######################################################
    #######################################################################
    # def visit_list(self, lst: tp.List) -> tp.List:
    #     return [self.visit(i) for i in lst]

    # def visit_tuple(self, tup: tp.Tuple) -> tp.Tuple:
    #     return tuple(self.visit_list(list(tup)))

    def visit_Constant(self, node: ast.Constant) -> Tuple[AST, Any]:
        if isinstance(node.value, (float, int, str)):
            return self.add_val(node, node.value), node.value
        elif node.value is None:
            return node, None
        else:
            raise AstInternalError(f'cannot recognize constant {node.value}')

    def visit_Tuple(self, node: ast.Tuple) -> Tuple[AST, Any]:
        vals = [self.visit(lo) for lo in node.elts]
        node.elts = [v[0] for v in vals]
        values = tuple(v[1] for v in vals)

        new_node = self.add_val(node, values)
        if isinstance(node.ctx, ast.Load):
            return new_node, values
        else:
            return new_node, None

    def visit_List(self, node: ast.List) -> Tuple[AST, Any]:
        vals = [self.visit(lo) for lo in node.elts]
        node.elts = [v[0] for v in vals]
        values = tuple(v[1] for v in vals)

        new_node = self.add_val(node, values)
        if isinstance(node.ctx, ast.Load):
            return new_node, values
        else:
            return new_node, None

    def visit_arg(self, node: ast.arg) -> Tuple[AST, Any]:
        actual_value = self.state.get_identifer(node.arg)
        # types = {type(actual_value)}
        if node.annotation is not None:
            # actual_type = type(actual_value)
            # node.args[i].annotation = ast.Name(quote_green(str(actual_type)))
            # types.add(node.annotation)
            node.annotation = None
        return self.add_val(node, actual_value), actual_value

    def visit_BinOp(self, node: ast.BinOp) -> Tuple[AST, Any]:
        node.left, lhs = self.visit(node.left)
        node.right, rhs = self.visit(node.right)

        var_left = self.state.get_unique_local_var("INTERNAL_BINOP_LHS")
        var_right = self.state.get_unique_local_var("INTERNAL_BINOP_RHS")
        self.state.set_identifier(var_left, lhs)
        self.state.set_identifier(var_right, rhs)

        fake_ast = ast.BinOp(ast.Name(var_left), node.op, ast.Name(var_right))
        eval_str = astunparse.unparse(fake_ast)
        try:
            val = eval(eval_str, self.state.globals, self.state.locals)
        except Exception as e:
            self.state.locals.pop(var_left)
            self.state.locals.pop(var_right)
            raise AstRunTimeError(f"failed to eval {eval_str} in Binop due to {e}")
        self.state.locals.pop(var_left)
        self.state.locals.pop(var_right)

        return self.add_val(node, val), val

    def visit_Compare(self, node: ast.Compare) -> Tuple[AST, Any]:
        node.left, lhs = self.visit(node.left)
        fake_left = self.state.get_unique_local_var("INTERNAL_COMPARATOR_LEFTVAR")
        self.state.locals[fake_left] = lhs
        fake_variables = []
        for i in range(len(node.comparators)):
            variable = self.state.get_unique_local_var(f"INTERNAL_COMPARATOR_VAR{i}")
            node.comparators[i], self.state.locals[variable] = self.visit(node.comparators[i])
            fake_variables.append(variable)
        fake_comparators = [ast.Name(id, ctx=ast.Load()) for id in fake_variables]
        fake_ast = ast.Compare(left=ast.Name(fake_left, ctx=ast.Load()), ops=node.ops, comparators=fake_comparators)
        eval_str = astunparse.unparse(fake_ast)

        def remove_fake_vars():
            for name in fake_variables:
                self.state.locals.pop(name)
            self.state.locals.pop(fake_left)

        try:
            val = eval(eval_str, self.state.globals, self.state.locals)
        except Exception as e:
            remove_fake_vars()
            raise AstRunTimeError(f"failed to eval {eval_str} in Compare due to {e}")
        remove_fake_vars()
        return self.add_val(node, val), val

    def visit_BoolOp(self, node: ast.BoolOp) -> Tuple[AST, Any]:
        nvals = [self.visit(v) for v in node.values]
        node.values = [v[0] for v in nvals]
        vals = [v[1] for v in nvals]
        if isinstance(node.op, ast.And()):
            val = True
            for v in vals:
                val = val and v
        elif isinstance(node.op, ast.Or()):
            val = False
            for v in vals:
                val = val or v
        return self.add_val(node, val), val

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Tuple[AST, Any]:
        node.operand, val = self.visit(node.operand)
        variable = self.state.get_unique_local_var("INTERNAL_UNIOPVAR")
        self.state.locals[variable] = val
        fake_ast = ast.UnaryOp(node.op, ast.Name(variable))
        eval_str = astunparse.unparse(fake_ast)
        try:
            val = eval(eval_str, self.state.globals, self.state.locals)
        except Exception as e:
            self.state.locals.pop(variable)
            raise AstRunTimeError(f"failed to eval {eval_str} due to {e}")
        self.state.locals.pop(variable)
        return self.add_val(node, val), val

    def visit_Name(self, node: ast.Name) -> Tuple[AST, Any]:
        if node.id == 'super' and isinstance(node.ctx, ast.Load):
            # this case shouldn't be necessary, but for some reason, the debugger fails
            #   to fetch 'super' from the global name space
            return node, super
        val = self.state.get_identifer(node.id)
        new_node = self.add_val(node, val)
        if isinstance(node.ctx, ast.Load):
            return new_node, val
        else:
            return new_node, val

    def visit_Attribute(self, node: ast.Attribute) -> Tuple[AST, Any]:
        # TODO: handle @property
        node.value, val = self.visit(node.value)

        # extract the @property object without calling it
        type_val = val if inspect.isclass(val) else type(val)
        if hasattr(type_val, node.attr):
            prop = getattr(type_val, node.attr)
            if isinstance(prop, property):
                fn_obj = prop.fget
                t_node = self.trace.get_trace(fn_obj)
                if t_node is not None or self.should_trace_prompt(fn_obj):
                    ret = self.trace_class_meth(fn_obj, type_val, args=[val])
                    return self.add_val(node, ret), ret

        # unfortunately, hasattr calls the property fn object
        if hasattr(val, node.attr):
            val = getattr(val, node.attr)
            new_node = self.add_val(node, val)
            if isinstance(node.ctx, ast.Load):
                return new_node, val
            else:
                return new_node, None
        else:
            raise AstRunTimeError(f"Cannot get attribute {node.attr} from {astunparse.unparse(node.value)}")

    def visit_Call(self, node: ast.Call) -> Tuple[AST, Any]:
        """
        Tracing rules:
            0. If the object doesn't satisfy self.trace.can_trace and default_should_trace,
                then we don't trace (has to be hashable, has attribute __name__, does not belong to torch, etc.)
            1. If the object is already in self.trace, then we always trace
            2. If the object is in self.trace_blacklist, then we don't trace
            3. We ask for user input, if input is anything other than 'n', we trace,
                if it is 'n', we add object to the blacklist, and don't trace
        """
        node.func, obj_callable = self.visit(node.func)
        arg_values = []
        for i in range(len(node.args)):
            if isinstance(node.args[i], ast.Starred):
                node.args[i].value, val = self.visit(node.args[i].value)
                arg_values.extend(val)
            else:
                node.args[i], val = self.visit(node.args[i])
                arg_values.append(val)

        kwarg_values = {}
        for i in range(len(node.keywords)):
            node.keywords[i].value, val = self.visit(node.keywords[i].value)
            kwarg_values[node.keywords[i].arg] = val

        def call_directly():
            try:
                # all supers should have been converted at this point
                # from super().meth to super(Class, self).meth
                if obj_callable == super:
                    assert len(arg_values) > 0, "super not properly converted"
                return obj_callable(*arg_values, **kwarg_values)
            except Exception as e:
                raise AstRunTimeError(str(e))

        def handle_base_trace_call(obj_callable, arg_values, kwarg_values, parent_cls=None):
            """
            This function handles if we should trace any subsequent calls
            parent_cls is supplied because I don't have a way of getting the class of any
            objects of the form 'MyClass.foo', as its not a bound method
            """
            sub_node = self.trace.get_trace(obj_callable)
            if sub_node is not None:
                return self.trace_code_obj(obj_callable, arg_values, kwarg_values)
            if parent_cls is not None:
                # could be a static method of a class
                if self.should_trace_prompt(
                    parent_cls, f"should trace parent class {parent_cls} of method {obj_callable}? [y]/n"
                ):
                    return self.trace_class_meth(obj_callable, parent_cls, arg_values, kwarg_values)
                else:
                    return call_directly()
            if self.should_trace_prompt(obj_callable):
                return self.trace_code_obj(obj_callable, arg_values, kwarg_values)
            return call_directly()

        def handle_trace_call():
            if not (self.trace.can_trace(obj_callable) and default_should_trace(obj_callable)):
                return call_directly()
            print('--> ' + vis_typed_ast(node))
            # unfortunately, super(MyClass, self).foo.__self__ == MyClass
            #   I have no way of extracting the superclass of MyClass from foo
            #   there is unbounded recursion if we don't have this check
            if isinstance(node.func, ast.Attribute):
                parent_val = getattr(node.func.value, TYPEDNODE_ATTR)
                assert isinstance(parent_val, AstTypedNode)
                if isinstance(parent_val.value, super):
                    inherited_cls = type(obj_callable.__self__)
                    for base in inherited_cls.__bases__:
                        # simulate object orientated method dispatch
                        base_cls = get_class_of_method(obj_callable, base)
                        if base_cls is not None:
                            meth = getattr(base_cls, obj_callable.__name__)
                            return handle_base_trace_call(
                                meth, arg_values=[obj_callable.__self__] + arg_values, kwarg_values=kwarg_values
                            )
                    return call_directly()
            parent_cls = None
            if isinstance(node.func, ast.Attribute) and inspect.isfunction(obj_callable):
                # could be a static method of a class
                parent_val = getattr(node.func.value, TYPEDNODE_ATTR)
                assert isinstance(parent_val, AstTypedNode)
                if not inspect.isclass(parent_val.value):
                    parent_cls = type(parent_val.value)
                else:
                    parent_cls = parent_val.value
            return handle_base_trace_call(obj_callable, arg_values, kwarg_values, parent_cls)

        func_return = handle_trace_call()
        return self.add_val(node, func_return), func_return

    #######################################################################
    ### Statements ########################################################
    #######################################################################

    def simulate_assign(self, expr: ast.Expr, val: Any):
        """Binds whatever is val to expr, which is the left hand side of the assignment"""

        variable = self.state.get_unique_local_var("INTERNAL_NODE_ASSIGN_VAR")
        # we just want to execute assignment statement
        self.state.locals[variable] = val

        fake_ast = []
        # handle globals
        if len(self.state.marked_global[-1]) > 1:
            fake_ast.append(ast.Global([v for v in self.state.marked_global[-1]]))
        # do not assume that the lower asts are typed
        fake_ast.append(ast.Assign(targets=[expr], value=ast.Name(id=variable, ctx=ast.Load())))
        exec_str = astunparse.unparse(fake_ast)
        # print(exec_str)
        try:
            exec(exec_str, self.state.globals, self.state.locals)
        except Exception as e:
            self.state.locals.pop(variable)
            raise AstRunTimeError(f"Cannot execute Assign: {exec_str} due to {e}")
        self.state.locals.pop(variable)

    def visit_Assign(self, node: ast.Assign) -> Tuple[AST, Any]:
        node.value, val = self.visit(node.value)

        if len(node.targets) > 1:
            self.simulate_assign(ast.Tuple(elts=node.targets, ctx=ast.Store()), val=val)
        else:
            self.simulate_assign(node.targets[0], val=val)

        # annotate targets
        node.targets, _ = self.visit(node.targets)

        return node, None

    def visit_arguments(self, node: ast.arguments) -> Tuple[AST, Any]:
        for i, arg in enumerate(node.args):
            node.args[i], _ = self.visit(arg)

        for i, arg in enumerate(node.kwonlyargs):
            node.kwonlyargs[i], _ = self.visit(arg)

        return node, None

    def visit_Break(self, node: ast.Break) -> Tuple[AST, Any]:
        self.inner_break = True
        return node, None

    def visit_stmt(self, stmt: ast.stmt) -> Tuple[AST, bool]:
        """
        This function should not be reached by self.visit,
        handles warnings and exceptions thrown on the line/statement level
        """
        if isinstance(stmt, ast.FunctionDef):
            self.state.add_error("We don't support internal nested functions yet")
        else:
            try:
                stmt, _ = self.visit(stmt)
            except AstRunTimeError as e:
                self.state.add_error(str(e))

        success = len(self.state.errors) == 0
        return self.state.add_errors(self.state.add_warnings(stmt)), success

    def visit_stmt_body(self, stmt_lst: tp.List[ast.stmt]) -> Tuple[AST, bool]:
        """This function should not be reached by self.visit"""
        success = True
        for i, stmt in enumerate(stmt_lst):
            if self.returned:
                break
            self.trace_msg('---> ' + vis_typed_ast(stmt))
            stmt_lst[i], succ = self.visit_stmt(stmt)
            success &= succ
            if self.inner_break and not success:
                break
        # success indicates if we should early break when executing a block of statements
        return stmt_lst, success

    def visit_If(self, node: ast.If) -> Tuple[AST, Any]:
        try:
            node.test, cond = self.visit(node.test)
        except Exception as e:
            self.state.add_error(str(e))
            return node, None

        self.trace_msg("tracing if " + vis_typed_ast(node.test))
        self.indent()
        if cond:
            node.body, success = self.visit_stmt_body(node.body)
        else:
            node.orelse, success = self.visit_stmt_body(node.orelse)
        if not success:
            self.state.add_error("if stmt error")
        self.unindent()
        return node, None

    def visit_For(self, node: ast.For) -> Tuple[AST, Any]:
        try:
            node.iter, xiter = self.visit(node.iter)
        except AstInternalError as e:
            raise e
        # except AstRunTimeError as e:
        #     self.state.add_error(str(e))

        self.trace_msg("tracing for " + vis_typed_ast(node.iter))
        self.indent()
        xiter = iter(xiter)
        execute_orelse = True
        while True:
            try:
                val = next(xiter)
                self.simulate_assign(node.target, val)
                node.body, success = self.visit_stmt_body(node.body)
                if self.returned:
                    execute_orelse = False
                    break
                if not success:
                    execute_orelse = False
                    break
                if self.inner_break:
                    self.inner_break = False
                    execute_orelse = False
                    break
            except StopIteration:
                break
        if not success:
            self.state.add_error("for stmt error")
            return node, None
        if execute_orelse:
            self.trace_msg("tracing orelse")
            self.indent()
            node.orelse, success = self.visit_stmt_body(node.orelse)
            self.unindent()
        if not success:
            self.state.add_error("for stmt orelse error")
        self.unindent()
        return node, None

    def visit_Pass(self, node: ast.Pass) -> Tuple[AST, Any]:
        return node, None

    def visit_Return(self, node: ast.Return) -> Tuple[AST, Any]:
        self.returned = True
        if node.value is not None:
            node.value, val = self.visit(node.value)
            self.state.set_return(val)
            return node, val
        else:
            self.state.set_return(None)
            return node, None

    def visit_Expr(self, node: ast.Expr) -> Tuple[AST, Any]:
        node.value, _ = self.visit(node.value)
        return node, None

    def visit_Module(self, node: ast.Module) -> Tuple[AST, Any]:
        if len(node.body) > 1:
            self.state.add_error("error, multiple defined bodies")
        node.body[0], _ = self.visit(node.body[0])

        return self.state.add_errors(self.state.add_warnings(node)), None

    def visit_Global(self, node: ast.Global) -> Tuple[AST, Any]:
        self.state.mark_globals(node)
        return node, None

    def trace_function(self, code_obj, args=[], kwargs={}) -> Any:
        assert inspect.isfunction(code_obj), "expected code_obj of trace_function to be a function"
        self.trace.add_trace(code_obj)  # this is a no op when code_obj is already in the trace
        node = self.trace.get_trace(code_obj)
        assert isinstance(node, ast.FunctionDef)
        # update locals to have the argument names defined
        self.state.push_stack(node.args, args, kwargs)
        self.state.globals.update(code_obj.__globals__)

        try:
            # annotate args
            node.args, _ = self.visit(node.args)
        except AstRunTimeError as e:
            self.state.add_error(str(e))
            self.state.pop_stack()
            return self.state.add_errors(self.state.add_warnings(node)), None

        node.body = self.visit_stmt_body(node.body)[0]
        success = self.trace.set_trace(code_obj, node)
        if not success:
            raise AstRunTimeError("cannot set trace for trace_function")

        val = self.state.pop_stack()
        # return barrier is at the function level
        self.returned = False
        return val

    def add_trace_class(self, cls_obj):
        assert inspect.isclass(cls_obj)
        self.trace.add_trace(cls_obj)
        node: ast.ClassDef = self.trace.get_trace(cls_obj)
        node = RewriteSuper().visit(node)
        for i, (base_node, base_obj) in enumerate(zip(node.bases, cls_obj.__bases__)):
            node.bases[i] = self.add_val(base_node, base_obj)
        self.trace.set_trace(cls_obj, node)

    def trace_bases(self, cls_obj, indent_level=0):
        assert inspect.isclass(cls_obj)
        bases = list(filter(lambda x: default_should_trace(x), cls_obj.__bases__))
        if len(bases) > 0:
            print(" " * indent_level + f"class {cls_obj.__name__} inherits from:")
            indent_level += 1
            for base in bases:
                if self.trace.get_trace(base) is None:
                    cond = self.should_trace_prompt(
                        base, msg=" " * indent_level + f"{base.__name__}, trace this class?"
                    )
                    if cond:
                        self.add_trace_class(base)
                        self.trace_bases(base, indent_level)
                    else:
                        self.trace_blacklist.add(base)

    def trace_class(self, code_obj, args=[], kwargs={}):
        """This traces the class constructor"""
        assert inspect.isclass(code_obj)
        self.add_trace_class(code_obj)
        self.trace_bases(code_obj)
        # self.trace_subclass(code_obj) # implement some kind of strategy for tracing subclasses

        selves = code_obj.__new__(code_obj)
        if self.trace.get_trace(getattr(code_obj, '__init__')) is None:
            # this means that either the current class does not contain __init__
            #   or one of the super classes contains __init__, but the user chose not to trace
            return selves.__init__(*args, **kwargs)
        self.trace_function(getattr(code_obj, '__init__'), [selves] + args, kwargs)
        return selves

    def trace_class_meth(self, meth, cls_obj, args=[], kwargs={}):
        assert inspect.isfunction(meth)
        assert inspect.isclass(cls_obj)
        assert hasattr(cls_obj, meth.__name__)
        self.add_trace_class(cls_obj)
        return self.trace_function(meth, args=args, kwargs=kwargs)

    def trace_bound_method(self, code_obj, args=[], kwargs={}):
        assert is_bound_method(code_obj)
        parent_cls = type(code_obj.__self__)
        actual_cls = get_class_of_method(code_obj, parent_cls)
        unbound_meth = getattr(actual_cls, code_obj.__name__)
        if unbound_meth != getattr(parent_cls, code_obj.__name__):
            raise AstInternalError(f"Origin class of method {code_obj} mismatched")
        return self.trace_class_meth(unbound_meth, actual_cls, args=[code_obj.__self__] + args, kwargs=kwargs)

    def trace_code_obj(self, code_obj, args=[], kwargs={}) -> Any:
        """
        expects code_obj to be one of the following cases
            1. a class, eg. inspect.isclass(code_obj) == True
            2. a bound method of a class
            3. a function
        """
        if not self.trace.can_trace(code_obj) or not default_should_trace(code_obj):
            raise AstRunTimeError("code_obj cannot be traced")
        if inspect.isfunction(code_obj):
            return self.trace_function(code_obj, args, kwargs)
        elif inspect.isclass(code_obj):
            return self.trace_class(code_obj, args, kwargs)
        elif is_bound_method(code_obj):
            return self.trace_bound_method(code_obj, args, kwargs)
        else:
            raise AstRunTimeError(f"code_obj {code_obj} is not detectd to be a function, bound method, or class")

    def __call__(self, code_obj, args=[], kwargs={}) -> Any:
        return self.trace_code_obj(code_obj, args, kwargs)

    def should_trace_prompt(self, obj: Any, msg=None) -> bool:
        if self.trace.can_trace(obj) and default_should_trace(obj):
            if obj in self.trace_blacklist:
                return False
            if msg is None:
                msg = f"should trace {obj.__name__}?"
            msg = ' ' * self.indent_level + (msg) + ' [y]/n '
            res = input(msg)
            if res == 'n':
                self.trace_blacklist.add(obj)
                return False
            else:
                return True
        else:
            return False


def get_single_expr_str(node: ast.AST) -> str:
    lhs_assign_str = astunparse.unparse(node).split("\n")
    assert len(lhs_assign_str) == 2 and lhs_assign_str[1] == ''
    lhs_assign_str = lhs_assign_str[0]
    return lhs_assign_str


def iter_fields(node):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def eliminate_indent(source: str) -> tp.Tuple[str, int]:
    lines = source.split('\n')
    indent = len(source)
    for line in lines:
        if len(line.strip()) == 0:
            continue
        indent = min(indent, len(line) - len(line.lstrip()))
    source = '\n'.join([line[indent:] for line in lines])
    return source, indent


def is_torch_module(code_obj) -> bool:
    if code_obj is torch.nn.Module:
        return True
    if inspect.isclass(code_obj):
        for b in code_obj.__bases__:
            if is_torch_module(b):
                return True
    return False


class VisualizeTypedAst(Unparser):
    def write_err_msg(self, tree: AST, attr: str, quote_fn, header: str):
        if hasattr(tree, attr):
            warnings = getattr(tree, attr)
            if len(warnings) == 0:
                self.fill(quote_fn(f"# {header}: {warnings[0]}"))
            else:
                self.fill(quote_fn(f"# {header}s:"))
                for i in warnings:
                    self.fill(quote_fn("#   " + i))

    def dispatch(self, tree):
        self.write_err_msg(tree, STMT_WARNING_ATTR, quote_warning, "WARNING")
        self.write_err_msg(tree, STMT_ERROR_ATTR, quote_fail, "ERROR")
        super().dispatch(tree)
        if hasattr(tree, TYPEDNODE_ATTR):
            types = getattr(tree, TYPEDNODE_ATTR)
            assert isinstance(types, AstTypedNode)
            type_repr = []
            for i in types.type:
                if callable(i) and hasattr(i, "__name__"):
                    type_repr.append(i.__name__)
                else:
                    type_repr.append(str(i)[8:-2])
            if len(types.type) == 1:
                self.write(quote_cyan(" :") + quote_green(type_repr[0] + ' '))
            elif len(types.type) > 1:
                self.write(quote_cyan(" :") + quote_green('Union[' + ', '.join(type_repr) + '] '))


class UnparseAstWithComments(Unparser):
    def write_err_msg(self, tree: AST, attr: str, header: str):
        if hasattr(tree, attr):
            warnings = getattr(tree, attr)
            if len(warnings) == 0:
                self.fill((f"# {header}: {warnings[0]}"))
            else:
                self.fill((f"# {header}s:"))
                for i in warnings:
                    self.fill(("#   " + i))

    def dispatch(self, tree):
        self.write_err_msg(tree, STMT_WARNING_ATTR, "WARNING")
        self.write_err_msg(tree, STMT_ERROR_ATTR, "ERROR")
        super().dispatch(tree)


def vis_typed_ast(tree):
    import six

    v = six.moves.cStringIO()
    VisualizeTypedAst(tree, file=v)
    return v.getvalue()


def vis_ast(tree):
    import six

    v = six.moves.cStringIO()
    UnparseAstWithComments(tree, file=v)
    return v.getvalue()


def dump_ast(code: str):
    print(ast.dump(ast.parse(code), indent=4))


def is_torch_path(name: str) -> bool:
    name = name.split(".")
    if len(name) > 0:
        return name[0] == "torch"
    return False


def belong_to_torch(code_obj) -> bool:
    belongs = False
    if hasattr(code_obj, "__module__") and code_obj.__module__ is not None:
        belongs |= is_torch_path(code_obj.__module__)
    if not belongs and hasattr(code_obj, "__package__") and code_obj.__package__ is not None:
        belongs |= is_torch_path(code_obj.__package__)
    return belongs


class StaticTypeInferPass(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.fn_globals = {}


class InferImports(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports = set()
        self.warnings = []

    def add_warnings(self, node: AST):
        if len(self.warnings) > 0:
            self.warnings, warnings = [], self.warnings
            if not hasattr(node, STMT_WARNING_ATTR):
                setattr(node, STMT_WARNING_ATTR, [])
            getattr(node, STMT_WARNING_ATTR).extend(warnings)
        return node

    def visit(self, node: AST):
        if isinstance(node, ast.stmt):
            self.add_warnings(node)
        return super().visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if hasattr(node, TYPEDNODE_ATTR):
            type_node = getattr(node, TYPEDNODE_ATTR)
            assert isinstance(type_node, AstTypedNode)
            val = type_node.value
            module_name = None

            if isinstance(val, types.ModuleType):
                module_name = val.__name__
            elif hasattr(val, '__module__') and val.__module__ in ('__main__', 'builtins'):
                pass
            elif hasattr(val, '__module__') and (inspect.isfunction(val) or inspect.isclass(val)):
                module_name = val.__module__
            if module_name is not None:
                if node.id != module_name:
                    self.imports.add(f"import {module_name} as {node.id}")
                else:
                    self.imports.add(f"import {module_name}")
            elif inspect.isfunction(val) or inspect.isclass(val):
                self.warnings.append(f"cannot find import for identifier {node.id}")


class Transpiler:
    FN_ROOT = "hidet_fn"
    METHOD_ROOT = "hidet_meth"
    MODULE_ROOT = "hidet_nn"
    IMPORTS = [
        "import hidet",
        f"import hidet.graph.frontend.torch.register_functions as {FN_ROOT}",
        f"import hidet.graph.frontend.torch.register_methods as {METHOD_ROOT}",
        f"import hidet.graph.frontend.torch.register_modules as {MODULE_ROOT}",
        "from functools import partial",
    ]

    type_dispatch = {
        torch.float32: hidet.float32,
        torch.float: hidet.float32,
        torch.float64: hidet.float64,
        torch.double: hidet.float64,
        torch.float16: hidet.float16,
        torch.bfloat16: hidet.bfloat16,
        torch.half: hidet.float16,
        torch.uint8: hidet.uint8,
        torch.int8: hidet.int8,
        torch.int16: hidet.int16,
        torch.short: hidet.int16,
        torch.int32: hidet.int32,
        torch.int: hidet.int32,
        torch.int64: hidet.int64,
        torch.long: hidet.int64,
        torch.complex32: None,
        torch.complex64: hidet.complex64,
        torch.cfloat: None,
        torch.complex128: hidet.complex128,
        torch.cdouble: hidet.complex64,
        torch.quint8: None,
        torch.qint8: None,
        torch.qint32: None,
        torch.bool: hidet.boolean,
        torch.quint4x2: None,
        torch.quint2x4: None,
    }

    def __init__(self) -> None:
        self.errors = []
        for imp in self.IMPORTS:
            exec(imp)
        self.hidet_fns = set(
            {
                fn
                for _, fn in eval(f"{self.FN_ROOT}.__dict__").items()
                if hasattr(fn, "__hash__") and fn.__hash__ is not None
            }
        )
        self.hidet_meths = set(
            {
                fn
                for _, fn in eval(f"{self.METHOD_ROOT}.__dict__").items()
                if hasattr(fn, "__hash__") and fn.__hash__ is not None
            }
        )
        self.hidet_mods = set(
            {
                fn
                for _, fn in eval(f"{self.MODULE_ROOT}.__dict__").items()
                if hasattr(fn, "__hash__") and fn.__hash__ is not None
            }
        )
        self.warnings = []
        self.errors = []

    def replacement_fn_attr(self, name: str, root):
        attr = ast.Attribute()
        attr.value = ast.Name(id=root, ctx=ast.Load())
        attr.attr = name
        attr.ctx = ast.Load()
        return attr

    def get_obj_expr(self, hidet_obj) -> AST:
        if hidet_obj in self.hidet_fns:
            return self.replacement_fn_attr(name=hidet_obj.__name__, root=self.FN_ROOT)
        if hidet_obj in self.hidet_meths:
            return self.replacement_fn_attr(name=hidet_obj.__name__, root=self.METHOD_ROOT)
        if hidet_obj in self.hidet_mods:
            return self.replacement_fn_attr(name=hidet_obj.__name__, root=self.MODULE_ROOT)
        raise AstInternalError(f"cannot find {hidet_obj.__name__} in caches")

    def visit(self, node: AST) -> AST:
        if hasattr(node, TYPEDNODE_ATTR):
            type = getattr(node, TYPEDNODE_ATTR)
            assert isinstance(type, AstTypedNode)
            method = 'visit_' + node.__class__.__name__
            if not hasattr(self, method):
                return self.generic_visit(node)
            visitor = getattr(self, method)
            node = visitor(node, type.type)
        else:
            node = self.generic_visit(node)
        if isinstance(node, (ast.stmt, ast.Module)):
            return self.add_errors(self.add_warnings(node))
        else:
            return node

    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        if not isinstance(value, AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def add_errors(self, node: AST) -> AST:
        assert isinstance(
            node, (ast.stmt, ast.mod)
        ), f"cannot add warnings to a non-statment ast node, adding {type(node)}"
        if len(self.errors) > 0:
            self.errors, errors = [], self.errors
            setattr(node, STMT_ERROR_ATTR, errors)
        return node

    def add_warnings(self, node: AST) -> AST:
        assert isinstance(
            node, (ast.stmt, ast.mod)
        ), f"cannot add warnings to a non-statment ast node, adding {type(node)}"
        if len(self.warnings) > 0:
            self.warnings, warnings = [], self.warnings
            setattr(node, STMT_WARNING_ATTR, warnings)
        return node

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_error(self, msg: str):
        self.errors.append(msg)

    def visit_stmt(self, node: AST) -> AST:
        node = self.visit(node)
        return self.add_errors(self.add_warnings(node))

    def handle_torch_types(self, obj) -> tp.Union[AST, None]:
        if obj == torch.dtype or isinstance(obj, torch.dtype):
            if obj in self.type_dispatch:
                if self.type_dispatch[obj] is not None:
                    type_str = str(self.type_dispatch[obj]).split(".")[1]
                    return self.replacement_fn_attr(type_str, root="hidet")
                else:
                    self.errors.append(f"found no equivalent type {obj} in hidet")
            else:
                self.errors.append("obj is of instance dtype but not in dispatch table")
        return None

    def get_type(self, node: AST) -> Any:
        if not hasattr(node, TYPEDNODE_ATTR):
            return None
        inner_types = getattr(node, TYPEDNODE_ATTR)
        assert isinstance(inner_types, AstTypedNode)
        inner_types = inner_types.type
        if len(inner_types) > 1:
            self.errors.append(f"type instability for inner attr.value {inner_types}")
        return list(inner_types)[0]

    def visit_Attribute(self, node: ast.Attribute, types: tp.Set):
        if isinstance(node.ctx, ast.Store):
            return node
        if len(types) > 1:
            self.add_warning(f"type instability, {types} detected for node attribute {node.attr}")
        obj = list(types)[0]
        if callable(obj):
            if get_hidet_function(obj) is not None:
                fn = get_hidet_function(obj)
                if len(fn.functions) > 1:
                    self.add_warning(f"multiple overloaded hidet functions detected for {obj}, attr {node.attr}")
                fn = fn.functions[0]
                replacement = self.get_obj_expr(fn)
                return replacement
            elif inspect.isfunction(obj) and belong_to_torch(obj):
                self.add_warning(
                    f"detected {obj} if a function and belongs to torch, but is not found in hidet function registry"
                )
                return node
            if get_hidet_method(obj) is not None:
                meth = get_hidet_method(obj)
                if len(meth.functions) > 1:
                    self.add_warning(f"multiple overloaded hidet methods detected for {obj}, attr {node.attr}")
                meth = meth.functions[0]
                replacement = ast.Call(
                    func=ast.Name("partial", ctx=ast.Load()),
                    args=[self.get_obj_expr(meth), self.visit(node.value)],
                    keywords=[],
                )
                return replacement
            elif (
                isinstance(self.get_type(node.value), torch.Tensor)
                and hasattr(obj, '__name__')
                and hasattr(torch.Tensor, obj.__name__)
            ):
                if hasattr(hidet.Tensor, obj.__name__):
                    self.add_warning(
                        f"assuming that it is correct to replace attribute {obj.__name__} with hidet's version"
                    )
                else:
                    self.add_warning(
                        f"detected {obj.__name__} is a method of torch.Tensor, \
 but is not found in hidet method registry and is not an attribute of hidet.Tensor"
                    )
                return node
            if obj == torch.nn.Module:
                return ast.parse("hidet.nn.Module").body[0].value
            if obj == torch.nn.ModuleList:
                return ast.parse("hidet.nn.ModuleList").body[0].value
            if get_hidet_module(obj) is not None:
                mod = get_hidet_module(obj)
                replacement = self.get_obj_expr(mod)
                return replacement
            elif inspect.isclass(obj) and belong_to_torch(obj):
                self.add_warning(
                    f"detected that {obj.__name__} belongs to torch, but is not found in hidet module registry"
                )
                return node

        type_rep = self.handle_torch_types(obj)
        if type_rep is not None:
            return type_rep

        if hasattr(node.value, TYPEDNODE_ATTR):
            inner_type = self.get_type(node.value)
            if belong_to_torch(inner_type):
                if inner_type == torch.Tensor:
                    # this makes the assumption that all torch values will be replaced by hidet values
                    if not hasattr(hidet.Tensor, node.attr):
                        self.errors.append(
                            f"detected that torch.Tensor has attr {node.attr}, but hidet.Tensor does not"
                        )
                    else:
                        self.errors.append(
                            f"making implicit assumption that its correct to \
                                replace attribute {node.attr} (from torch) with hidet's version"
                        )
                        return node
                else:
                    self.errors.append(
                        f"detected that type of attr.value ({inner_type}) \
                            belongs to torch, but found no hidet equivalent"
                    )
        if belong_to_torch(obj):
            self.errors.append(f"detected that {obj} of attr {node.attr} belongs to torch, but cannot find equivalent")
        node.value = self.visit(node.value)
        return node

    def visit_Name(self, node: ast.Name, types: tp.Set):
        if len(types) > 1:
            self.errors.append(f"type instability, {types} detected for node Name {node.id}")
        obj = list(types)[0]
        if callable(obj):
            if get_hidet_function(obj) is not None:
                fn = get_hidet_function(obj)
                if len(fn.functions) > 1:
                    self.errors.append(
                        f"warning: multiple overloaded hidet functions detected for {obj}, name {node.id}"
                    )
                fn = fn.functions[0]
                replacement = self.replacement_fn_attr(fn.__name__, self.FN_ROOT)
                return replacement
        type_rep = self.handle_torch_types(obj)
        if type_rep is not None:
            return type_rep
        if obj != torch.Tensor and belong_to_torch(obj):
            self.errors.append(
                f"warning: detected that {obj} of Name {node.id} belongs to torch, but cannot find equivalent"
            )
        return node

    def visit_arg(self, node: ast.arg, types: tp.Set):
        if len(types) > 1:
            self.errors.append(f"warning: type instability, {types} detected for node arg {node.arg}")
        obj = list(types)[0]
        type_rep = self.handle_torch_types(obj)
        if type_rep is not None:
            node.annotation = type_rep
        if obj == torch.Tensor:
            node.annotation = self.replacement_fn_attr("Tensor", "hidet")
        if hasattr(obj, '__module__') and obj.__module__ == "builtins":
            node.annotation = ast.Name(str(obj)[8:-2], ctx=ast.Load())
        return node


def get_hidet_function(fn):
    from hidet.graph.frontend.torch.interpreter import Registry

    if fn in Registry.registered_functions:
        return Registry.registered_functions[fn]
    return None


def get_hidet_method(fn):
    from hidet.graph.frontend.torch.interpreter import Registry

    if fn in Registry.registered_methods:
        return Registry.registered_methods[fn]
    return None


def get_hidet_module(mod):
    from hidet.graph.frontend.torch.interpreter import Registry

    if mod in Registry.registered_modules:
        return Registry.registered_modules[mod]
    return None


def visualize(func, args, kwargs={}, transpile=True):
    interpreter = AstInterpreter()
    interpreter(func, args, kwargs)
    for _, ast_node in interpreter.trace.traced.items():
        print(vis_typed_ast(ast_node))
        if transpile:
            transpiled = Transpiler().visit(ast_node)
            print(vis_typed_ast(transpiled))


def vis_interpreter(interpreter: AstInterpreter):
    for _, ast_node in interpreter.trace.traced.items():
        print("Interpreted AST")
        print(vis_typed_ast(ast_node))
        transpiled = Transpiler().visit(ast_node)
        print("Transpiled AST")
        print(vis_typed_ast(transpiled))


def transpiled_str(interpreter: AstInterpreter):
    body = []
    imports = InferImports()
    for _, ast_node in interpreter.trace.traced.items():
        transpiled = Transpiler().visit(ast_node)
        imports.visit(transpiled)
        body.append(transpiled)

    imported = list(imports.imports)
    imported.extend(Transpiler.IMPORTS)

    mod = ast.Module(body=body)
    res = vis_ast(mod)
    res = '\n'.join(imported) + res
    return res
