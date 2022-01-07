from typing import Dict, List, Union
from hidet.ir.node import Node
from hidet.ir.type import BaseType, FuncType
from hidet.ir.expr import Var
from hidet.ir.stmt import Stmt


class IRModule(Node):
    def __init__(self, funcs):
        self.functions: Dict[str, Union[Function, FunctionGroup]] = funcs
        self.global_vars: Dict[str, Var] = {}

    def include(self, module):
        for name, func in module.functions.items():
            assert name not in self.functions
            self.functions[name] = func

        for name, var in module.global_vars.items():
            self.global_vars[name] = var

    def lookup(self, name):
        return self.functions[name]

    def lookup_var(self, name):
        assert name in self.functions
        if name not in self.global_vars:
            func = self.functions[name]
            self.global_vars[name] = Var(name, FuncType.from_func(func))
        return self.global_vars[name]


class FunctionGroup(Node):
    def __init__(self, group):
        self.group: List[Function] = group


class Function(Node):
    valid_attrs = [
        'worker'
    ]
    """
    Valid Attrs:
     'worker': one of 'host', 'grid', 'threadblock', 'warp', 'thread'
        the worker to run the function.
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
        if default:
            return default
        raise AttributeError(f'{attr_name} not found')


