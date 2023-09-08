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
from __future__ import annotations

import ast as py_ast
import inspect
from types import FunctionType
from typing import Tuple, Optional, List, Any, Dict

from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.type import FuncType, BaseType, func_type
from hidet.lang.transpiler import PythonToHidetTranslator
from hidet.runtime.compiled_module import CompiledModule


class ScriptFunction:
    def __init__(self, func: Function):
        self.func: Function = func
        self.callees: List[ScriptFunction] = []

    def __call__(self, *args):
        pass


def eliminate_indent(source: str) -> Tuple[str, int]:
    lines = source.split('\n')
    indent = len(source)
    for line in lines:
        if len(line.strip()) == 0:
            continue
        indent = min(indent, len(line) - len(line.lstrip()))
    source = '\n'.join([line[indent:] for line in lines])
    return source, indent


def eliminate_decorators(source: str) -> Tuple[str, int]:
    lines = source.split('\n')
    num_decorators = 0
    for line in lines:
        if len(line) > 0 and line[0] == '@':
            num_decorators += 1
        else:
            break
    source = '\n'.join(lines[num_decorators:])
    return source, num_decorators


def script(func: FunctionType) -> Function:
    """
    Decorator to convert a Python function to a Hidet function.

    Parameters
    ----------
    func: FunctionType
        The python function to be converted to a Hidet function.

    Returns
    -------
    ret: Function
        The hidet.ir.Function that is converted from the given Python function.
    """
    # Extract the source code of given function
    lines, start_line = inspect.getsourcelines(func)
    file = inspect.getsourcefile(func)
    source = ''.join(lines)
    source, col_offset = eliminate_indent(source)
    source, inc_lineno = eliminate_decorators(source)
    start_line += inc_lineno
    parsed: py_ast.AST = py_ast.parse(source=source)

    # Get the environment (globals and binding of free variables)
    # See the data model of python for the details of func.__globals__, func.__closure__ and func.__code__:
    #     https://docs.python.org/3/reference/datamodel.html
    env: Dict[str, Any] = func.__globals__.copy()
    func_freevar_names: List[str] = list(func.__code__.co_freevars)
    func_freevar_cells: List[Any] = [v.cell_contents for v in func.__closure__] if func.__closure__ else []
    assert len(func_freevar_names) == len(func_freevar_cells)
    env.update(dict(zip(func_freevar_names, func_freevar_cells)))

    # get the type annotations of function parameters.
    func_annotations: Dict[str, Any] = func.__annotations__

    # Translate the Python function into Hidet function
    translator = PythonToHidetTranslator(
        file=file, start_lineno=start_line, start_column=col_offset, env=env, func_annotations=func_annotations
    )
    hidet_function = translator(parsed)

    # add function to current script module if we are in a script module context
    ctx = ScriptModuleContext.current_context()
    if ctx:
        ctx.append_function(hidet_function)
    assert isinstance(hidet_function, Function)
    return hidet_function


class ScriptModuleContext:
    contexts: List[ScriptModuleContext] = []

    def __init__(self):
        self.name2var: Dict[str, Var] = {}
        self.functions: List[Function] = []
        self.extern_functions: Dict[str, Var] = {}

    def __enter__(self):
        self.contexts.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.contexts.pop()

    @staticmethod
    def current_context() -> Optional[ScriptModuleContext]:
        contexts = ScriptModuleContext.contexts
        return contexts[-1] if len(contexts) > 0 else None

    def append_function(self, function: Function):
        self.functions.append(function)
        self.name2var[function.name] = Var(hint=None, type=FuncType.from_func(function), name=function.name)

    def lookup(self, name: str) -> Optional[Var]:
        if name not in self.name2var:
            return None
        return self.name2var[name]

    def define_global_var(self, name: str, var_type: BaseType) -> Var:
        if name in self.name2var:
            raise ValueError(f'Global variable {name} is already defined.')
        self.name2var[name] = Var(hint=None, type=var_type, name=name)
        return self.name2var[name]

    def declare_extern_func(self, name: str, param_types, ret_type):
        if name in self.extern_functions:
            raise ValueError(f'Extern function {name} is already declared.')
        self.extern_functions[name] = Var(hint=None, name=name, type=func_type(param_types, ret_type))
        return self.extern_functions[name]

    def ir_module(self) -> IRModule:
        return IRModule(
            functions={func.name: func for func in self.functions},
            global_vars=self.name2var,
            extern_functions=self.extern_functions,
        )

    def build(self) -> CompiledModule:
        return self.ir_module().build()


def script_module() -> ScriptModuleContext:
    return ScriptModuleContext()
