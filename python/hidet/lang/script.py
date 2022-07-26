from typing import Callable, Tuple, Optional, List, Any, Dict
from types import FunctionType
import ast as py_ast
import inspect
from hidet.ir.func import Function
from .transpiler import PythonToHidetTranslator


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
    # Extract the source code of given function
    lines, start_line = inspect.getsourcelines(func)
    file = inspect.getsourcefile(func)
    source = ''.join(lines)
    source, col_offset = eliminate_indent(source)
    source, inc_lineno = eliminate_decorators(source)
    start_line += inc_lineno
    parsed: py_ast.AST = py_ast.parse(source=source)

    # Get the environment (binding of free variables)
    # See the data model of python for the details of func.__closure__ and func.__code__:
    #     https://docs.python.org/3/reference/datamodel.html
    func_freevar_names: List[str] = list(func.__code__.co_freevars)
    func_freevar_cells: List[Any] = [v.cell_contents for v in func.__closure__] if func.__closure__ else []
    assert len(func_freevar_names) == len(func_freevar_cells)
    env: Dict[str, Any] = {name: value for name, value in zip(func_freevar_names, func_freevar_cells)}
    func_annotations: Dict[str, Any] = func.__annotations__

    # Translate the Python function into Hidet function
    translator = PythonToHidetTranslator(
        file=file,
        start_lineno=start_line,
        start_column=col_offset,
        env=env,
        func_annotations=func_annotations
    )
    hidet_function = translator(parsed)
    return hidet_function

