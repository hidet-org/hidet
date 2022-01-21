from typing import Dict, List, Union, Optional
from hidet.ir.node import Node
from hidet.ir.type import BaseType, FuncType
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.expr import Var
from hidet.ir.stmt import Stmt
from hidet.ir.task import Task


class Function(Node):
    valid_attrs = [
        'worker',
        'packed_func',
        'label'
    ]
    """
    Valid Attrs:
     'worker': Union[Host, Grid, ThreadBlock, Warp, Thread]
        the worker to run the function.
     'packed_func': Function
        the target function that this packed_func has packed
     'label': str
        the label of this function when it is in a function group
    """

    def __init__(self, name, params, body, ret_type, local_vars, attrs=None):
        self.name = name
        self.params: List[Var] = params
        self.body: Stmt = body
        self.ret_type: BaseType = ret_type
        self.local_vars: List[Var] = local_vars
        self.attrs = attrs if attrs else {}

        assert all(attr in self.valid_attrs for attr in self.attrs)

    def annotate(self, attr_name, attr_value, update=False):
        assert attr_name in self.valid_attrs
        if attr_name in self.attrs and not update:
            raise AttributeError(f'{attr_name} has existed')
        self.attrs[attr_name] = attr_value

    def get_attr(self, attr_name, default=None):
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        return default


class FunctionGroup(Node):
    def __init__(self, name, group=None):
        self.name = name
        self.group: List[Function] = group if group else []

    def append(self, func: Function):
        assert func.name == self.name
        self.group.append(func)


class IRModule(Node):
    def __init__(self, funcs=None, task=None):
        self.task: Optional[Task] = task
        self.functions: Dict[str, Union[Function, FunctionGroup]] = funcs if funcs else {}
        self.global_vars: Dict[str, Var] = {}

    def include(self, module):
        for name, func in module.functions.items():
            if name in self.functions:
                funcs = [func] if isinstance(func, Function) else list(func.group)
                if isinstance(self.functions[name], FunctionGroup):
                    self.functions[name].group.extend(funcs)
                else:
                    self.functions[name] = FunctionGroup(name, funcs + [self.functions[name]])
            else:
                self.functions[name] = func

        for name, var in module.global_vars.items():
            self.global_vars[name] = var

    def lookup(self, name_or_var: Union[str, Var]):
        if isinstance(name_or_var, Var):
            name = name_or_var.hint
        else:
            name = name_or_var
        return self.functions[name]

    def lookup_var(self, name):
        assert name in self.functions
        if name not in self.global_vars:
            func = self.functions[name]
            if isinstance(func, Function):
                self.global_vars[name] = Var(name, FuncType.from_func(func))
            elif isinstance(func, FunctionGroup):
                self.global_vars[name] = Var(name, FuncType.from_func(func.group[0]))
            else:
                raise ValueError()

        return self.global_vars[name]

    def add(self, name, func: Union[Function, FunctionGroup]):
        if name in self.functions:
            existed_func = self.functions[name]
            if isinstance(existed_func, FunctionGroup):
                existed_func.group.append(func)
            else:
                self.functions[name] = FunctionGroup(name, group=[existed_func, func])
        else:
            self.functions[name] = func


